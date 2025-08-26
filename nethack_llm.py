from collections import deque
import re
import json
import yndf
from nle import nethack
import gymnasium as gym
from yndf.dict_state import get_status_dict
from pprint import pprint
from llm import LLMWithTools
from typing import Tuple

from yndf.wrapper_actions import COORDINATE_TO_ACTION

def _print(env, action, status):
    print("\033[2J\033[H", end="")
    pprint(status)
    if action is not None:
        for key, value in action.items():
            print(key, ":", value)
    env.render()

def _get_direction(action):
    match action['direction']:
        case 'north':
            return nethack.CompassDirection.N
        case 'south':
            return nethack.CompassDirection.S
        case 'east':
            return nethack.CompassDirection.E
        case 'west':
            return nethack.CompassDirection.W
        case 'northeast':
            return nethack.CompassDirection.NE
        case 'northwest':
            return nethack.CompassDirection.NW
        case 'southeast':
            return nethack.CompassDirection.SE
        case 'southwest':
            return nethack.CompassDirection.SW

def _get_actions(env, action, state):
    result = []
    actions = env.unwrapped.actions

    if (count := action.get('count', 1)) > 1:
        result.append(actions.index(ord('n')))
        for s in str(count):
            result.append(actions.index(ord(s)))

    match action['action']:
        case 'eat':
            index = actions.index(nethack.Command.EAT)

        case 'attack':
            index = actions.index(_get_direction(action))

        case 'move':
            index = actions.index(_get_direction(action))

        case _:
            index = actions.index(nethack.MiscDirection.WAIT)

    result.append(index)

    if o := action.get('object'):
        for letter, name in state.player.inventory.items():
            if o in name:
                result.append(actions.index(ord(letter)))
                break

    return result

def _make_history(msg_hist, action_hist):
    result = {}
    if msg_hist:
        result['messages'] = [f"time:{timestamp} {msg}" for msg, timestamp in msg_hist]

    if action_hist:
        actions = []
        for timestamp, action in action_hist:
            action = _get_action_history_item(action)
            action['time'] = timestamp
            actions.append(action)

        result['actions'] = actions

    return result

def _get_action_history_item(action):
    result = action.copy()
    if "explanation" in result:
        del result['explanation']
    if "remember" in result:
        del result['remember']
    if "forget" in result:
        del result['forget']

    return result

NAME_TO_ACTION = {
    'n' : nethack.CompassDirection.N,
    's' : nethack.CompassDirection.S,
    'e' : nethack.CompassDirection.E,
    'w' : nethack.CompassDirection.W,
    'nw' : nethack.CompassDirection.NW,
    'sw' : nethack.CompassDirection.SW,
    'ne' : nethack.CompassDirection.NE,
    'se' : nethack.CompassDirection.SE
}

class NethackLLMContext:
    def __init__(self, env, state : yndf.NethackState):
        self.initial_state = state
        self.env = env

        self._steps = []

    def append(self, step, allow_repeat: bool):
        if not allow_repeat and self.has_stepped:
            raise ValueError("Can only call one NETHACK-ACTION.")

        self._steps.append(step)

    def throw_if_repeat(self):
        if self.has_stepped:
            raise ValueError("Can only call one NETHACK-ACTION.")

    @property
    def done(self):
        return self._steps and any(s.terminated or s.truncated for s in self._steps)

    @property
    def has_stepped(self):
        return bool(self._steps)

    @property
    def current_state(self):
        return self._steps[-1].state if self._steps else self.initial_state

    @property
    def steps(self):
        return self._steps

class NethackLLMStep:
    def __init__(self, action, name, params, obs, reward, terminated, truncated, info, state):
        self.action = action
        self.function_name = name
        self.params = params
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.state = state

class NethackTools:
    _success_message = "Completed successfully.  Return the requested OUTPUT json to the user for the next step."

    def __init__(self, llm : LLMWithTools, env, state, allow_expert : bool):
        self._curr = NethackLLMContext(env, state)

        def ask_an_expert(question: str, game_state: str) -> str:
            """Ask a Nethack expert a question and return the answer.

            Args:
                question: The natural-language question to ask the expert.
                game_state: The current state of the game, which is a GAME_STATE_DICT.

            Returns:
                str: The expert's response to the question.
            """
            msg = "\n".join([question, "GAME_STATE", game_state])
            _, result = self._expert.chat(msg)
            return result

        def move(direction: str) -> str:
            """NETHACK-ACTION: Move in the specified direction.

            Args:
                direction: n s e w nw sw ne se

            Returns:
                str: A description of whether the action was successful.
            """
            action = NAME_TO_ACTION.get(direction)
            if action is None:
                raise ValueError(f"Invalid direction: {direction}")

            step = self._step(action, "move", (direction,))
            return self._get_success_response(step)

        def search(num_turns : int) -> str:
            """NETHACK-ACTION: Search for hidden objects or passages in the 8 tiles around the current one.

            Args:
                num_turns:
                    The number of turns to search for hidden objects or passages.  Most experts recommend searching
                    for 22 total turns on the same square for a high likelihood of success.

            Returns:
                str: A description of whether the action was successful.
            """
            step = self._step(nethack.Command.SEARCH, "search", (num_turns,))
            return self._get_success_response(step)

        def kick(direction: str) -> str:
            """NETHACK-ACTION: Kick in the specified direction.  Kick can break open locked doors and chests.  Kick can
            be used to kick monsters, and kick objects on the floor (usually trying to hit a monster with it).  Kicking
            a wall will harm the player.

            Args:
                direction: n s e w nw sw ne se

            Returns:
                str: A description of whether the action was successful.
            """
            dir_action = NAME_TO_ACTION.get(direction)
            if dir_action is None:
                raise ValueError(f"Invalid direction: {direction}")

            step = self._step(nethack.Command.KICK, "kick", (direction,))
            if "what direction" in step.state.message.lower():
                step = self._step(dir_action, "kick_direction", (direction,))
                return self._get_success_response(step)

            return "Tried to perform kick action but did not receive the expected response."

        def wait(num_turns: int) -> str:
            """NETHACK-ACTION: Wait for a specified number of turns.  Waiting for one turn is used to reposition a
            monster by waiting for it to use.  Waiting can also be used to restore health at the cost
            of hunger, when waiting to restore health typically num_turns should be 50 at a time.  This will
            automatically stop waiting if a monster is spotted, or health reaches full.  It will not waste time when
            used in this way.

            Args:
                num_turns: The number of turns to wait.

            Returns:
                str: A description of whether the action was successful.
            """
            if num_turns > 1:
                self._curr.env.unwrapped.nethack.step('n')
                for c in str(num_turns):
                    self._curr.env.unwrapped.nethack.step(c)

            step = self._step(nethack.Command.WAIT, "wait", (num_turns,))
            return self._get_success_response(step)

        def eat(inventory_id: str) -> str:
            """NETHACK-ACTION: Eat a food item from the inventory or the floor.

            Args:
                inventory_id: The ID of the food item to eat or 'floor'.

            Returns:
                str: A description of whether the action was successful.
            """
            step = self._step(nethack.Command.EAT, "eat", (inventory_id,))
            if self._is_yn_question(step.state.message) and inventory_id == "floor":
                step = self._step(ord('y'), "eat", ("floor",))

            if "What do you want to eat?" in step.state.message:
                step = self._step(ord(inventory_id), "eat", (inventory_id,))

            return self._get_success_response(step)

        def respond_yn(response: str) -> str:
            """NETHACK-ACTION: Respond to a yes/no question.

            Args:
                response: The response to the question ('y', 'n', or 'q').

            Returns:
                str: A description of whether the action was successful.
            """
            if len(response) != 1 or response not in "ynq":
                raise ValueError(f"Invalid response: {response}")

            step = self._step(ord(response), "respond_yn", (response,))
            return self._get_success_response(step)

        def respond_direction(direction: str) -> str:
            """NETHACK-ACTION: Respond to a direction question.

            Args:
                direction: The direction to respond with: n, s, e, w, nw, sw, ne, se, here.

            Returns:
                str: A description of whether the action was successful.
            """
            if direction == "here":
                value = ord('s')
            elif direction in NAME_TO_ACTION:
                value = NAME_TO_ACTION[direction]
            else:
                raise ValueError(f"Invalid direction: {direction}")

            step = self._step(value, "respond_direction", (direction,))
            return self._get_success_response(step)

        def respond_inventory(inventory_id: str) -> str:
            """NETHACK-ACTION: Respond to an inventory question.

            Args:
                inventory_id: The ID of the inventory item to respond with.

            Returns:
                str: A description of whether the action was successful.
            """
            if len(inventory_id) != 1:
                raise ValueError(f"Invalid inventory ID: {inventory_id}")

            step = self._step(ord(inventory_id), "respond_inventory", (inventory_id,))
            return self._get_success_response(step)

        if allow_expert:
            self._expert = LLMWithTools(system_prompt="Answer all Nethack questions with confidence",
                                        max_new_tokens=1024*128, temperature=0.3, model=llm.model,
                                        tokenizer=llm.tokenizer, io_lock=llm._gen_lock)
            llm.register_tool("ask_an_expert", ask_an_expert)

        llm.register_tool("move", move)
        llm.register_tool("search", search)
        llm.register_tool("kick", kick)
        llm.register_tool("wait", wait)
        llm.register_tool("eat", eat)
        llm.register_tool("respond_inventory", respond_inventory)
        llm.register_tool("respond_yn", respond_yn)
        llm.register_tool("respond_direction", respond_direction)

    def _is_yn_question(self, msg):
        return "[" in msg and re.search(r'\[[ynq\?]\]', msg)

    def _get_success_response(self, step : NethackLLMStep) -> str:
        msg = step.state.message
        if "in what direction" in msg.lower():
            return f"The game responded to this action with the following prompt, please call respond_direction:\n{msg}"

        if "[" in msg:
            match = re.search(r'\[([^\]]+)\ or .*\*.*]', msg)
            if match:
                result = "The game responded to this action with the following prompt, please call respond_inventory:"
                result += f"\n{msg}"
                return result

            #match yn
            match = re.search(r'\[[ynq\?]\]', msg)
            if match:
                return f"The game responded to this action with the following prompt, please call respond_yn:\n{msg}"

        return self._success_message

    def _step(self, action, name, params):
        if self._curr is None:
            raise ValueError("No current context.")

        if self._curr.done:
            raise ValueError("Episode is already terminated.")

        index = self.env.unwrapped.actions.index(action)
        obs, reward, terminated, truncated, info = self.env.step(index)
        state = yndf.NethackState(obs, info, self.env, self.current_state)
        step = NethackLLMStep(action, name, params, obs, reward, terminated, truncated, info, state)
        self._curr.append(step, False)
        return step

    def begin_step(self):
        self._curr = NethackLLMContext(self._curr.env, self._curr.current_state)

    def end_step(self):
        steps = self._curr.steps
        if not steps:
            raise ValueError("No NETHACK-ACTION was executed.")

        self._curr = None
        return steps

def main():
    with open("instructions.txt", "r", encoding="utf-8") as f:
        instructions = f.read()

    llm = LLMWithTools(system_prompt=instructions, max_new_tokens=1024*8, temperature=0.3)

    env = gym.make("NetHackChallenge-v0")
    o, i = env.reset()

    state = yndf.NethackState(o, i, env)
    nethack_tools = NethackTools(llm, env, state, False)

    msg_hist = deque(maxlen=5)
    action_hist = deque(maxlen=10)
    status = get_status_dict(state)
    memory = [None] * 25
    status['history'] = _make_history(msg_hist, action_hist)

    _print(env, None, status)

    while True:
        nethack_tools.begin_step()
        js = json.dumps(status)
        thinking, action = llm.chat(js)
        nethack_tools.end_step()
        action = json.loads(action)
        print("\n--- THINKING ---\n", thinking)
        print("\n--- RESULT ---\n", json.dumps(action, indent=2))

        if "remember" in action:
            for i, v in enumerate(memory):
                if not v:
                    memory[i] = action["remember"]
                    break
        if "forget" in action:
            i = int(action["forget"])
            memory[i] = None

        steps = _get_actions(env, action, state)
        if not steps:
            print("No actions")
            continue

        for step in steps:
            prev_message = ""
            more_loop = True
            while more_loop:
                o, _, term, trunc, i = env.step(step)
                if term or trunc:
                    return

                state = yndf.NethackState(env, o, i, state)
                if more_loop := "--More--" in state.message:
                    prev_message += state.message.split("--More--")[0] + ' '

        status = get_status_dict(state)
        if prev_message:
            status['message'] = prev_message + state.message

        action_hist.appendleft((state.time, action))
        status['history'] = _make_history(msg_hist, action_hist)
        _print(env, action, status)

        if 'message' in status and status['message']:
            msg_hist.appendleft((status['message'], state.time))

main()
