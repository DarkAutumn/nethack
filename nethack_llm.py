from collections import deque
from typing import Callable
from pprint import pprint

from pydantic import Field, conint, constr
from llm.gpt5_nethack_agent import GPT5NanoAgent, JsonStep, LLMStep, StepKind
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

class NethackStep:
    def __init__(self, action, name, params, obs, reward, terminated, truncated, info, state, count):
        self.action = action
        self.count = count
        self.function_name = name
        self.params = params
        self.obs = obs
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.state = state
        self.message = state.message

class NethackContext:
    def __init__(self, env, start_state):
        self.env = env
        self.state = start_state
        self.steps = []
        self._callbacks = []

    def register_callback(self, callback : Callable[['NethackContext', NethackStep], None]):
        self._callbacks.append(callback)

    def step(self, actions, name, params, *, count = None):
        if self.done:
            raise ValueError("Episode is already terminated.")

        if count:
            self.env.unwrapped.nethack.step(ord('n'))
            for c in str(count):
                self.env.unwrapped.nethack.step(ord(c))

        if not isinstance(actions, list):
            actions = [actions]

        for action in actions:
            step = self._perform_one_action(name, params, count, action)
            while step.state.waiting_for_more:
                step = self._perform_one_action(name, params, count, nethack.MiscAction.MORE, step.message)

        return step

    def _perform_one_action(self, name, params, count, action, message_prefix = None):
        index = self.env.unwrapped.actions.index(action if not isinstance(action, str) else ord(action))
        obs, reward, terminated, truncated, info = self.env.step(index)
        self.state = yndf.NethackState(obs, info, self.env, self.state)
        step = NethackStep(action, name, params, obs, reward, terminated, truncated, info, self.state, count)
        self.steps.append(step)
        if message_prefix:
            step.message = f"{message_prefix} {self.state.message}"
        else:
            step.message = self.state.message

        for callback in self._callbacks:
            callback(self, step)
        return step

    @property
    def done(self):
        """Returns whether the episode is done."""
        if not self.steps:
            return False
        return self.steps[-1].terminated or self.steps[-1].truncated

class NethackTools:
    _success_message = "Completed successfully.  Return the requested OUTPUT json to the user for the next step."

    def __init__(self, context : NethackContext):
        self.context = context

    def move(self, direction: str, far: bool):
        """Move in the specified direction."""
        action = NAME_TO_ACTION.get(direction)
        if action is None:
            raise ValueError(f"Invalid direction: {direction}")

        if far:
            self.context.step(nethack.Command.MOVEFAR, "move", ("far",))

        self.context.step(action, "move", (direction,))

    def search(self, num_turns : int):
        """Search for hidden objects or passages in the 8 tiles around the current one."""
        count = num_turns if num_turns > 1 else None
        self.context.step(nethack.Command.SEARCH, "search", (num_turns,), count=count)

    def kick(self, direction: str):
        """Kick in the specified direction."""
        dir_action = NAME_TO_ACTION.get(direction)
        if dir_action is None:
            raise ValueError(f"Invalid direction: {direction}")

        continue_kicks = True

        while continue_kicks:
            step = self.context.step(nethack.Command.KICK, "kick", (direction,))
            if "what direction" in step.message.lower():
                step = self.context.step(dir_action, "kick_direction", (direction,))
                continue_kicks = "WHAMMM!!!" in step.message and not self.context.state.floor.enemies.any()
            else:
                continue_kicks = False

    def ascend(self):
        """Ascend to the previous floor."""
        self.context.step(nethack.MiscDirection.UP, "ascend", None)

    def descend(self):
        """Descend to the next floor."""
        self.context.step(nethack.MiscDirection.DOWN, "descend", None)

    def fire(self, direction: str):
        """Fire a projectile in the specified direction."""
        dir_action = NAME_TO_ACTION.get(direction)
        if dir_action is None:
            raise ValueError(f"Invalid direction: {direction}")

        step = self.context.step(nethack.Command.FIRE, "fire", (direction,))
        if "what direction" in step.message.lower():
            step = self.context.step(dir_action, "fire_direction", (direction,))

    def throw(self, inventory_id: str, direction: str):
        """Throw an item from the inventory in the specified direction."""
        dir_action = NAME_TO_ACTION.get(direction)
        if dir_action is None:
            raise ValueError(f"Invalid direction: {direction}")

        step = self.context.step([nethack.Command.THROW, ord(inventory_id)], "throw", (inventory_id, direction))
        if "What do you want to throw?" in step.message:
            step = self.context.step(ord(inventory_id), "throw - select", (inventory_id,))
            if "In what direction?" in step.message:
                step = self.context.step(dir_action, "throw_direction", (inventory_id, direction))

    def wait(self, num_turns: int):
        """Wait for a specified number of turns."""

        count = num_turns if num_turns > 1 else None
        self.context.step(nethack.MiscDirection.WAIT, "wait", (num_turns,), count=count)

    def eat(self, inventory_id: str):
        """Eat a food item from the inventory or the floor."""
        step = self.context.step(nethack.Command.EAT, "eat", (inventory_id,))
        while "eat it?" in step.message:
            step = self.context.step(ord('n'), "eat - confirm", (inventory_id,))

        self.context.step(ord(inventory_id), "eat - select", (inventory_id,))

    def eat_floor(self):
        """Eat a food item from the floor."""
        step = self.context.step(nethack.Command.EAT, "eat_floor", None)
        while "eat it?" in step.message:
            step = self.context.step(ord('y'), "eat_floor - confirm", None)

        if "What do you want to eat?" in step.message:
            step = self.context.step(nethack.Command.ESC, "eat_floor - select", None)

    def wear(self, inventory_id: str):
        """Wear armor from the inventory."""
        self.context.step([nethack.Command.WEAR, inventory_id], "wear", (inventory_id,))

    def wield(self, inventory_id: str):
        """Wield a weapon from the inventory."""
        self.context.step([nethack.Command.WIELD, inventory_id], "wield", (inventory_id,))

    def put_on(self, inventory_id: str):
        """Put on an item from the inventory."""
        self.context.step([nethack.Command.PUTON, inventory_id], "put_on", (inventory_id,))
        self.context.step(ord(inventory_id), "put_on", (inventory_id,))

    def take_off(self, inventory_id: str):
        """Take off an item from the inventory."""
        step = self.context.step(nethack.Command.TAKEOFF, "take_off", (inventory_id,))
        if "You finish taking" not in step.message:
            self.context.step(ord(inventory_id), "take_off - select", (inventory_id,))

    def quiver(self, inventory_id: str):
        """Quiver an item from the inventory."""
        self.context.step([nethack.Command.QUIVER, inventory_id], "quiver", (inventory_id,))

    def pick_up(self) -> str:
        """Pick up an item from the ground."""
        self.context.step(nethack.Command.PICKUP, "pick_up", None)

    def apply(self, inventory_id: str) -> str:
        """Apply an item from the inventory."""
        self.context.step([nethack.Command.APPLY, ord(inventory_id)], "apply", (inventory_id,))

    def drop(self, inventory_id: str) -> str:
        """Drop an item from the inventory."""
        self.context.step([nethack.Command.DROP, ord(inventory_id)], "drop", (inventory_id,))

    def respond(self, response: str) -> str:
        """Respond to a prompt."""
        for item in self._get_items(response):
            match item.lower():
                case "esc" | "escape":
                    step = self.context.step(nethack.Command.ESC, "respond", (item,))
                case " " | "space":
                    step = self.context.step(nethack.TextCharacters.SPACE, "respond", (item,))
                case "enter":
                    step = self.context.step(nethack.MiscAction.MORE, "respond", (item,))
                case _:
                    step = self.context.step(ord(item), "respond", (item,))

    def _get_items(self, response: str) -> list[str]:
        """Extract item names from a response string."""
        result = []
        i = 0
        while i < len(response):
            if response[i] == '[':
                i += 1
                start = i
                while i < len(response) and response[i] != ']':
                    i += 1
                result.append(response[start:i])
            else:
                result.append(response[i])

            i += 1

        return result

def _get_cost(tokens_used):
    PRICE_INPUT = 0.050 / 1_000_000
    PRICE_CACHED = 0.005 / 1_000_000
    PRICE_OUTPUT = 0.400 / 1_000_000

    input_tokens = tokens_used.get("input", 0)
    cached_tokens = tokens_used.get("cached", 0)
    output_tokens = tokens_used.get("output", 0)

    cost = (
        input_tokens * PRICE_INPUT
        + cached_tokens * PRICE_CACHED
        + output_tokens * PRICE_OUTPUT
    )
    return cost

def main():
    with open("instructions.txt", "r", encoding="utf-8") as f:
        instructions = f.read()
    agent = GPT5NanoAgent(
        instructions_text=instructions,
        model="gpt-5-nano",              # or "gpt-5-thinking-nano"
        reasoning_effort="medium",       # "minimal" | "medium" | "high"
        client=OpenAI(),
    )

    total_cost = 0.0

    messages = deque(maxlen=32)
    actions = deque(maxlen=16)

    env = gym.make("NetHackChallenge-v0")
    o, i = env.reset()

    context = NethackContext(env, yndf.NethackState(o, i, env))
    context.env.render()
    input("Press Enter to start...")

    def _on_llm_step(step : LLMStep):
        if step.kind == StepKind.OUTPUT_DELTA:
            if step.thinking:
                print(f"\033[33m{step.content}\033[0m", end="", flush=True)
            else:
                print(f"\033[34m{step.content}\033[0m", end="", flush=True)
        elif step.kind == StepKind.FUNCTION_CALL:
            print()
            print(f"\033[31m{step.function_name}({step.arguments})\033[0m", flush=True)
            print(step.content, flush=True)
        elif step.kind == StepKind.JSON:
            decision = step.content
            actions.appendleft(
                {
                    "time": context.state.time,
                    "action": decision.get("action", ""),
                    "args": decision.get("args", ""),
                    "notes": decision.get("notes", ""),
                })

    def _on_step(context, step : NethackStep):
        if step.message:
            messages.appendleft({"time": context.state.time, "message": step.message})

    context.register_callback(_on_step)
    nethack_tools = NethackTools(context)
    #register_tools(agent, nethack_tools)

    while not context.done:
        context.env.render()

        with open("instructions.txt", "r", encoding="utf-8") as f:
            agent.system_prompt = f.read()
        response, tokens = agent.chat(get_status_dict(context.state, messages, actions), output_callback=_on_llm_step)
        pprint(tokens)

        result = [x for x in response if isinstance(x, JsonStep)]
        if not result:
            raise ValueError("No valid JSON response found.")
        result = result[-1].content
        actions.appendleft(
            {
                "time": context.state.time,
                "action": result.get("action", ""),
                "args": result.get("args", ""),
                "notes": result.get("notes", ""),
            })

        tool = getattr(nethack_tools, result["action"])
        tool(**result.get("args", {}))

        total_cost += _get_cost(tokens)
        print()
        print(f"Total cost so far: ${total_cost:.6f}")
        print()

main()
