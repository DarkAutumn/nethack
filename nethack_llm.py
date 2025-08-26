from collections import deque
import sys
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


class NethackChatState:
    def __init__(self, env, state : yndf.NethackState):
        self.initial_state = state
        self.env = env

        self._steps = []

    def step(self, action):
        if self.done:
            raise ValueError("Episode is already terminated.")
        index = self.env.unwrapped.actions.index(action)
        obs, reward, terminated, truncated, info = self.env.step(index)
        state = yndf.NethackState(obs, info, self.env, self.current_state)
        self._steps.append((action, obs, reward, terminated, truncated, info, state))
        return self.done

    def throw_if_repeat(self):
        if self.has_stepped:
            raise ValueError("Can only call one NETHACK-ACTION.")

    def throw_if_not_called(self):
        if not self.has_stepped:
            raise ValueError("No NETHACK-ACTION was executed.")

    @property
    def done(self):
        return self._steps and any(s[3] or s[4] for s in self._steps)

    @property
    def has_stepped(self):
        return bool(self._steps)

    @property
    def current_state(self):
        return self._steps[-1][-1] if self._steps else self.initial_state

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

class NethackTools:
    _success_message = "Completed successfully.  Return the requested OUTPUT to the user for the next step."

    def __init__(self, llm : LLMWithTools, env, state, allow_expert : bool):
        self._chat_state = NethackChatState(env, state)

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
                str: A description of whether the move was successful.
            """
            action = NAME_TO_ACTION.get(direction)
            if action is None:
                raise ValueError(f"Invalid direction: {direction}")

            self._chat_state.step(action)
            return self._success_message

        if allow_expert:
            self._expert = LLMWithTools(system_prompt="Answer all Nethack questions with confidence",
                                        max_new_tokens=1024*128, temperature=0.3, model=llm.model,
                                        tokenizer=llm.tokenizer, io_lock=llm._gen_lock)
            llm.register_tool("ask_an_expert", ask_an_expert)

        llm.register_tool("move", move)

    def begin_step(self):
        self._chat_state = NethackChatState(self._chat_state.env, self._chat_state.current_state)

    def end_step(self):
        self._chat_state.throw_if_not_called()
        self._chat_state = None

def main():
    with open("instructions.txt", "r", encoding="utf-8") as f:
        instructions = f.read()

    llm = LLMWithTools(system_prompt=instructions, max_new_tokens=1024*128, temperature=0.3)

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
        thinking, action = llm.chat(json.dumps(status))
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
