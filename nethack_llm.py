from collections import deque
from typing import Literal, Callable
from pprint import pprint

from pydantic import BaseModel, Field, conint, constr
from llm.gpt5_nethack_agent import GPT5NanoAgent, LLMStep, JsonStep, StepKind
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

    def move(self, direction: str, far: bool) -> str:
        """Move in the specified direction."""
        action = NAME_TO_ACTION.get(direction)
        if action is None:
            raise ValueError(f"Invalid direction: {direction}")

        if far:
            self.context.step(nethack.Command.MOVEFAR, "move", ("far",))

        step = self.context.step(action, "move", (direction,))
        return self._get_llm_response(step, True)

    def search(self, num_turns : int) -> str:
        """Search for hidden objects or passages in the 8 tiles around the current one."""
        count = num_turns if num_turns > 1 else None
        step = self.context.step(nethack.Command.SEARCH, "search", (num_turns,), count=count)
        return self._get_llm_response(step, True)

    def kick(self, direction: str) -> str:
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

        return self._get_llm_response(step, True)

    def fire(self, direction: str) -> str:
        """Fire a projectile in the specified direction."""
        dir_action = NAME_TO_ACTION.get(direction)
        if dir_action is None:
            raise ValueError(f"Invalid direction: {direction}")

        step = self.context.step(nethack.Command.FIRE, "fire", (direction,))
        if "what direction" in step.message.lower():
            step = self.context.step(dir_action, "fire_direction", (direction,))

        return self._get_llm_response(step, True)

    def wait(self, num_turns: int) -> str:
        """Wait for a specified number of turns."""

        count = num_turns if num_turns > 1 else None
        step = self.context.step(nethack.MiscDirection.WAIT, "wait", (num_turns,), count=count)
        return self._get_llm_response(step, True)

    def eat(self) -> str:
        """Eat a food item from the inventory or the floor."""
        step = self.context.step(nethack.Command.EAT, "eat", None)
        return self._get_llm_response(step, True)

    def wield(self, inventory_id: str) -> str:
        """Wield a weapon from the inventory."""
        step = self.context.step([nethack.Command.WIELD, inventory_id], "wield", (inventory_id,))
        return self._get_llm_response(step, True)

    def wear(self, inventory_id: str) -> str:
        """Wear armor from the inventory."""
        step = self.context.step([nethack.Command.WEAR, inventory_id], "wear", (inventory_id,))
        return self._get_llm_response(step, True)

    def put_on(self, inventory_id: str) -> str:
        """Put on an item from the inventory."""
        step = self.context.step([nethack.Command.PUTON, inventory_id], "put_on", (inventory_id,))
        step = self.context.step(ord(inventory_id), "put_on", (inventory_id,))
        return self._get_llm_response(step, True)

    def take_off(self, inventory_id: str) -> str:
        """Take off an item from the inventory."""
        step = self.context.step([nethack.Command.TAKEOFF, inventory_id], "take_off", (inventory_id,))
        return self._get_llm_response(step, True)

    def quiver(self, inventory_id: str) -> str:
        """Quiver an item from the inventory."""
        step = self.context.step([nethack.Command.QUIVER, inventory_id], "quiver", (inventory_id,))
        return self._get_llm_response(step, True)

    def pick_up(self) -> str:
        """Pick up an item from the ground."""
        step = self.context.step(nethack.Command.PICKUP, "pick_up", None)
        return self._get_llm_response(step, True)


    def apply(self, inventory_id: str) -> str:
        """Apply an item from the inventory."""
        step = self.context.step([nethack.Command.APPLY, ord(inventory_id)], "apply", (inventory_id,))
        return self._get_llm_response(step, True)

    def drop(self, inventory_id: str) -> str:
        """Drop an item from the inventory."""
        step = self.context.step([nethack.Command.DROP, ord(inventory_id)], "drop", (inventory_id,))
        return self._get_llm_response(step, True)

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

        return self._get_llm_response(step, False)

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

    def _get_llm_response(self, step : NethackStep, is_action : bool) -> str:
        msg = step.message

        result = "Action completed. " if is_action else "Response sent. "

        if not msg:
            result += "Provide OUTPUT to the user."
        else:
            result += f"Nethack responded '{msg}'. "
            result += "If this is a prompt, use NETHACK-RESPONSE to respond, otherwise provide OUTPUT to the user."

        return result

# Reusable enums
DIRECTION_ENUM = ["n", "s", "e", "w", "nw", "ne", "sw", "se"]
DIRECTION_OR_HERE_ENUM = DIRECTION_ENUM + ["here"]

class MoveOrMelee(BaseModel):
    direction: Literal["n","s","e","w","nw","ne","sw","se"] = \
        Field(..., description="Compass direction to move/attack.")
    far: bool = Field(..., description="Whether the action is a far move (move until stopped).")

class Wait(BaseModel):
    num_turns: conint(ge=1, le=1000) = Field(..., description="Number of turns to wait")

class Search(BaseModel):
    num_turns: conint(ge=1, le=50) =  Field(..., description="Search turns (22 total is a common recommendation)")

class Kick(BaseModel):
    direction: Literal["n","s","e","w","nw","ne","sw","se"]

class Eat(BaseModel):
    pass

class PickUp(BaseModel):
    pass

class Wield(BaseModel):
    inventory_id : constr(min_length=1, max_length=1) = Field(..., description="Inventory id of the weapon to wield")

class Wear(BaseModel):
    inventory_id : constr(min_length=1, max_length=1) = Field(..., description="Inventory id of the item to wear")

class PutOn(BaseModel):
    inventory_id : constr(min_length=1, max_length=1) = Field(..., description="Inventory id of the accessory (ring or amulet) to put on")

class TakeOff(BaseModel):
    inventory_id : constr(min_length=1, max_length=1) = Field(..., description="Inven" \
    "tory id of the item to take off")

class Quiver(BaseModel):
    inventory_id : constr(min_length=1, max_length=1) = Field(..., description="Inventory id of the projectile to put into your quiver")

class Apply(BaseModel):
    inventory_id : constr(min_length=1, max_length=1) = Field(..., description="Inventory id of the item to apply")

class Fire(BaseModel):
    direction: Literal["n","s","e","w","nw","ne","sw","se"] = \
        Field(..., description="Compass direction to fire.")

class Respond(BaseModel):
    response: str = Field(..., description="Response to a prompt.  [ESC], [ENTER], or [SPACE] will be interpeted as those keys, otherwise this should be a single character.")

def register_tools(agent, tools : NethackTools):
    # pylint: disable=unnecessary-lambda

    agent.register_tool(MoveOrMelee, lambda direction, far: tools.move(direction, far),
                        "NETHACK-ACTION: Move the player or melee attack enemy if one is in the square you attempt to move into.  Use far=true to move in a direction until something happens (either you reach a wall or you spot an enemy or a status message occurs).  Prioritize far moves as much as possible as they are safe and cuts down on the total number of turns.")
    agent.register_tool(Wait, lambda num_turns: tools.wait(num_turns), "NETHACK-ACTION: Wait (rest) for the given number of turns.  num_turns==1 is useful for waiting on a monster to get closer, num_turns==50 is useful to wait for hp to regenerate.")
    agent.register_tool(Search, lambda num_turns: tools.search(num_turns), "NETHACK-ACTION: Search the 8 squares around you, a num_turns of 22 is common to ensure you fully found everything.  Search should only be used if you are standing on a 1.0 search score tile, otherwise you should move to objects or frontier instead of searching.")
    agent.register_tool(Kick, lambda direction: tools.kick(direction), "NETHACK-ACTION: Kick in a direction")
    agent.register_tool(Eat, lambda: tools.eat(),
                        "NETHACK-ACTION: Eat an item.  If there is something on the floor you will be prompted for a NETHACK-RESPONS y or n to eat that floor item, otherwise you will be prompted to use NETHACK-RESPONSE to select an inventory item.")
    agent.register_tool(Wield, lambda inventory_id: tools.wield(inventory_id),
                        "NETHACK-ACTION: Wield a weapon from your inventory.")
    agent.register_tool(Wear, lambda inventory_id: tools.wear(inventory_id),
                        "NETHACK-ACTION: Wear armor from your inventory.")
    agent.register_tool(PutOn, lambda inventory_id: tools.put_on(inventory_id),
                        "NETHACK-ACTION: Put on an accessory from your inventory (you can wear 2 rings and 1 amulet).")
    agent.register_tool(TakeOff, lambda inventory_id: tools.take_off(inventory_id),
                        "NETHACK-ACTION: Take off an item you are wearing.")
    agent.register_tool(Quiver, lambda inventory_id: tools.quiver(inventory_id),
                        "NETHACK-ACTION: Put a projectile into your quiver.")
    agent.register_tool(Apply, lambda inventory_id: tools.apply(inventory_id),
                        "NETHACK-ACTION: Apply or use an item from your inventory (lockpick, candle, etc).")
    agent.register_tool(Fire, lambda direction: tools.fire(direction),
                        "NETHACK-ACTION: Fire a projectile in a direction (make sure a bow is wielded if using arrows).")
    agent.register_tool(PickUp, lambda: tools.pick_up(),
                        "NETHACK-ACTION: Pick up an item from the ground.")
    agent.register_tool(Respond, lambda response: tools.respond(response),
                        "NETHACK-RESPONSE: Respond to an on screen prompt.")

def _print_output(step : LLMStep):
    if step.kind == StepKind.OUTPUT_DELTA:
        if step.thinking:
            print(f"\033[33m{step.content}\033[0m", end="", flush=True)
        else:
            print(f"\033[34m{step.content}\033[0m", end="", flush=True)
    elif step.kind == StepKind.FUNCTION_CALL:
        print()
        print(f"\033[31m{step.function_name}({step.arguments})\033[0m", flush=True)
        print(step.content, flush=True)

def _on_step(context, step):
    context.env.render()

def _get_result(chat_steps: list[LLMStep]):
    for step in chat_steps:
        if isinstance(step, JsonStep):
            return step.content

    raise ValueError("LLM did not produce a JsonStep")

def main():
    with open("instructions.txt", "r", encoding="utf-8") as f:
        instructions = f.read()
    agent = GPT5NanoAgent(
        instructions_text=instructions,
        model="gpt-5-nano",              # or "gpt-5-thinking-nano"
        reasoning_effort="medium",       # "minimal" | "medium" | "high"
        client=OpenAI(),
    )

    env = gym.make("NetHackChallenge-v0")
    o, i = env.reset()
    env.render()

    context = NethackContext(env, yndf.NethackState(o, i, env))
    context.register_callback(_on_step)
    nethack_tools = NethackTools(context)

    register_tools(agent, nethack_tools)

    while not context.done:
        start_time = context.state.time
        def is_complete():
            return context.state.time > start_time

        chat_steps, tokens = agent.chat(get_status_dict(context.state), is_complete, output_callback=_print_output)
        pprint(tokens)
        _get_result(chat_steps)

        # wait for "enter" on the keyboard
        input("Press Enter to continue...")

main()
