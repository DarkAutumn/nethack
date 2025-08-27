from collections import deque
import re
from typing import Literal
from pprint import pprint

from pydantic import BaseModel, Field, conint, constr
from llm.gpt5_nethack_agent import GPT5NanoAgent
from nle import nethack

from openai import OpenAI
import gymnasium as gym
from yndf.dict_state import get_status_dict

import yndf

def _print(env, action, status):
    #print("\033[2J\033[H", end="")
    pprint(status)
    if action is not None:
        for key, value in action.items():
            print(key, ":", value)
    env.render()

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

    def __init__(self, env, state):
        self._steps = None
        self._env = env
        self._state = state

    def __enter__(self):
        if self._steps is not None:
            raise ValueError("Already in a context.")

        self._steps = []
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._steps:
            self._state = self._steps[-1].state
        elif self._steps is None:
            raise ValueError("Not in a context.")

        self._steps = None

    def move(self, direction: str) -> str:
        """Move in the specified direction."""
        action = NAME_TO_ACTION.get(direction)
        if action is None:
            raise ValueError(f"Invalid direction: {direction}")

        step = self._step(action, "move", (direction,))
        return self._get_success_response(step)

    def search(self, num_turns : int) -> str:
        """Search for hidden objects or passages in the 8 tiles around the current one."""
        step = self._step(nethack.Command.SEARCH, "search", (num_turns,))
        return self._get_success_response(step)

    def kick(self, direction: str) -> str:
        """Kick in the specified direction."""
        dir_action = NAME_TO_ACTION.get(direction)
        if dir_action is None:
            raise ValueError(f"Invalid direction: {direction}")

        step = self._step(nethack.Command.KICK, "kick", (direction,))
        if "what direction" in step.state.message.lower():
            step = self._step(dir_action, "kick_direction", (direction,))
            return self._get_success_response(step)

        return "Tried to perform kick action but did not receive the expected response."

    def wait(self, num_turns: int) -> str:
        """Wait for a specified number of turns."""
        if num_turns > 1:
            self._env.unwrapped.nethack.step('n')
            for c in str(num_turns):
                self._env.unwrapped.nethack.step(c)

        step = self._step(nethack.Command.WAIT, "wait", (num_turns,))
        return self._get_success_response(step)

    def eat(self, inventory_id: str) -> str:
        """Eat a food item from the inventory or the floor."""
        step = self._step(nethack.Command.EAT, "eat", (inventory_id,))
        if self._is_yn_question(step.state.message) and inventory_id == "floor":
            step = self._step(ord('y'), "eat", ("floor",))

        if "What do you want to eat?" in step.state.message:
            step = self._step(ord(inventory_id), "eat", (inventory_id,))

        return self._get_success_response(step)

    def respond_yn(self, response: str) -> str:
        """Respond to a yes/no question."""
        if len(response) != 1 or response not in "ynq":
            raise ValueError(f"Invalid response: {response}")

        step = self._step(ord(response), "respond_yn", (response,))
        return self._get_success_response(step)

    def respond_direction(self, direction: str) -> str:
        """Respond to a direction question."""
        if direction == "here":
            value = ord('s')
        elif direction in NAME_TO_ACTION:
            value = NAME_TO_ACTION[direction]
        else:
            raise ValueError(f"Invalid direction: {direction}")

        step = self._step(value, "respond_direction", (direction,))
        return self._get_success_response(step)

    def respond_inventory(self, inventory_id: str) -> str:
        """Respond to an inventory question."""
        if len(inventory_id) != 1:
            raise ValueError(f"Invalid inventory ID: {inventory_id}")

        step = self._step(ord(inventory_id), "respond_inventory", (inventory_id,))
        return self._get_success_response(step)

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
        if self._steps is None:
            raise ValueError("Was not entered!")

        if self.done:
            raise ValueError("Episode is already terminated.")

        index = self._env.unwrapped.actions.index(action)
        obs, reward, terminated, truncated, info = self._env.step(index)
        state = yndf.NethackState(obs, info, self._env, self.prev_state)
        step = NethackLLMStep(action, name, params, obs, reward, terminated, truncated, info, state)
        self._steps.append(step)
        return step

    @property
    def done(self):
        return any(s.terminated or s.truncated for s in self._steps) if self._steps else False

    @property
    def steps(self):
        assert self._steps is not None, "Has not been entered!"
        return self._steps

    @property
    def prev_state(self):
        return self._steps[-1].state if self._steps else self._state

# Reusable enums
DIRECTION_ENUM = ["n", "s", "e", "w", "nw", "ne", "sw", "se"]
DIRECTION_OR_HERE_ENUM = DIRECTION_ENUM + ["here"]

class MOVE(BaseModel):
    direction: Literal["n","s","e","w","nw","ne","sw","se"] = Field(..., description="Compass direction to move/attack.")

class WAIT(BaseModel):
    num_turns: conint(ge=1, le=1000) = Field(..., description="Number of turns to wait")

class SEARCH(BaseModel):
    num_turns: conint(ge=1, le=50) =  Field(..., description="Search turns (22 total is a common recommendation)")

class KICK(BaseModel):
    direction: Literal["n","s","e","w","nw","ne","sw","se"]

class EAT(BaseModel):
    inventory_id: constr(min_length=1, max_length=5) = Field(..., description="Letter of item, or 'floor'")

class RespondYNArgs(BaseModel):
    response: Literal["y","n","q"]

class RespondDirectionArgs(BaseModel):
    direction: Literal["here","n","s","e","w","nw","ne","sw","se"]

class RespondInventoryArgs(BaseModel):
    inventory_id: constr(min_length=1, max_length=1)


def register_tools(agent, context):
    # pylint: disable=unnecessary-lambda

    agent.register_tool(MOVE, lambda direction: context.move(direction),
                        "NETHACK-ACTION: Move the player or melee attack enemy if one is in the square you attempt to move into.")
    agent.register_tool(WAIT, lambda num_turns: context.wait(num_turns), "NETHACK-ACTION: Wait (rest) for turns.")
    agent.register_tool(SEARCH, lambda num_turns: context.search(num_turns), "NETHACK-ACTION: Search around you.")
    agent.register_tool(KICK, lambda direction: context.kick(direction), "NETHACK-ACTION: Kick in a direction")
    agent.register_tool(EAT, lambda inventory_id: context.eat(inventory_id),
                        "NETHACK-ACTION: Eat inventory or floor food")
    agent.register_tool(RespondYNArgs, lambda response: context.respond_yn(response),
                        "NETHACK-RESPONSE: Answer a [ynq] prompt")
    agent.register_tool(RespondDirectionArgs, lambda direction: context.respond_direction(direction),
                        "NETHACK-RESPONSE: Answer a direction prompt")
    agent.register_tool(RespondInventoryArgs, lambda inventory_id: context.respond_inventory(inventory_id),
                        "NETHACK-RESPONSE: Answer an inventory prompt")


def main():
    with open("instructions.txt", "r", encoding="utf-8") as f:
        instructions = f.read()
    agent = GPT5NanoAgent(
        instructions_text=instructions,
        model="gpt-5-nano",              # or "gpt-5-thinking-nano"
        reasoning_effort="minimal",       # "minimal" | "medium" | "high"
        client=OpenAI(),
    )

    env = gym.make("NetHackChallenge-v0")
    o, i = env.reset()

    state = yndf.NethackState(o, i, env)
    nethack_tools = NethackTools(env, state)

    register_tools(agent, nethack_tools)

    msg_hist = deque(maxlen=5)
    action_hist = deque(maxlen=10)
    status = get_status_dict(state)
    memory = [None] * 25
    status['history'] = _make_history(msg_hist, action_hist)

    _print(env, None, status)

    while True:
        with nethack_tools:
            action = agent.play_nethack(status, print_stream=True)
            steps = nethack_tools.steps

        if "remember" in action:
            for i, v in enumerate(memory):
                if not v:
                    memory[i] = action["remember"]
                    break
        if "forget" in action:
            i = int(action["forget"])
            memory[i] = None

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
