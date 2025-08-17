# debugger.py
import argparse
import gymnasium as gym
import torch
from nle import nethack
from models import NethackPolicy
import yndf.gui
from yndf.wrapper_actions import DIRECTIONS, VERBS

from yndf.wrapper_actions import UserInputAction
from yndf.wrapper_rewards import NethackRewardWrapper
from ppo_train import ModelSaver


def get_action_masker(env: gym.Env) -> gym.Wrapper:
    action_masker = env
    while not hasattr(action_masker, "action_masks"):
        if isinstance(action_masker, gym.Wrapper):
            action_masker = action_masker.env
        else:
            raise ValueError("Environment does not support action masks.")
    return action_masker


def get_ending_handler(env: gym.Env):
    ending_handler = env
    while not isinstance(ending_handler, NethackRewardWrapper):
        if isinstance(ending_handler, gym.Wrapper):
            ending_handler = ending_handler.env
        else:
            raise ValueError("Environment does not support endings.")
    return ending_handler


class Controller(yndf.gui.NethackController):
    """A controller for the YenderFlow GUI debugger."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        self.model = None  # will be an InferenceAdapter
        self.obs = None

        self.action_masker = get_action_masker(env)
        endings = get_ending_handler(env).endings
        for x in endings:
            if x.name == "max-timesteps-reached":
                x.disable()

    def reset(self) -> yndf.NethackState:
        obs, info = self.env.reset()
        self.obs = obs
        return info["state"]

    def step(self, action: int | None = None) -> yndf.gui.StepInfo:
        action_mask = self.action_masker.action_masks()
        verb_mask, dir_mask = action_mask
        if action is None:
            if self.model is None:
                raise RuntimeError("No model set. Call set_model().")

            action, _ = self.model.predict(self.obs, deterministic=False, action_masks=action_mask,
                                           unsqueeze=True)
        else:
            action = UserInputAction(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs

        ending = info.get("ending", None)
        if ending is not None:
            assert terminated or truncated, "Episode should end if an ending is provided."

        action_mask = self.action_masker.action_masks()
        verb_mask, dir_mask = action_mask
        available_actions = []
        masked_actions = []

        for i, verb in enumerate(VERBS):
            if not verb_mask[i]:
                masked_actions.append(verb.name)
            else:
                if verb in (nethack.Command.MOVE, nethack.Command.KICK):
                    available_directions = [d.name for d in DIRECTIONS if dir_mask[i].any()]
                    available_actions.append(f"{verb.name}: {" ".join(available_directions)}")
                else:
                    available_actions.append(verb.name)

        state: yndf.NethackState = info["state"]
        properties = {
            "Actions": available_actions,
            "Disallowed": masked_actions,
            "Locked Doors": state.locked_doors,
        }

        if isinstance(action, UserInputAction):
            unwrapped_actions = self.env.unwrapped.actions
            action_name = unwrapped_actions[unwrapped_actions.index(action.action)].name
        else:
            if VERBS[action[0]] in (nethack.Command.MOVE, nethack.Command.KICK):
                action_name = f"{VERBS[action[0]].name}: {DIRECTIONS[action[1]].name}"
            else:
                action_name = VERBS[action[0]].name

        return yndf.gui.StepInfo(
            state, action, action_name, reward, list(info.get("rewards", {}).items()), properties, ending
        )

    def set_model(self, model_path: str) -> None:
        """
        Load a saved model (your custom checkpoint) and wrap it for inference.
        You provide how to build the policy from saved args + current env spaces.
        """
        def policy_builder(_: dict) -> torch.nn.Module:
            # pylint: disable=no-member
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return NethackPolicy(num_verbs=len(VERBS), glyph_vocab_size=nethack.NO_GLYPH).to(device)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ModelSaver.load_as_inference(model_path, policy_builder, device=device)


def main():
    parser = argparse.ArgumentParser(description="Run the YenderFlow GUI debugger.")
    parser.add_argument("model_path", help="Path to a trained model (your custom .zip/.pt).")
    args = parser.parse_args()

    env = gym.make("YenderFlow-v0", replay_dir="replays/")
    yndf.gui.run_gui(Controller(env), model_path=args.model_path)


if __name__ == "__main__":
    main()
