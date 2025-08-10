"""YenderFlow GUI Debugger for NetHack RL"""
import argparse
import gymnasium as gym
from sb3_contrib import MaskablePPO
import yndf.gui

from train import ACTIONS
from yndf.wrapper_actions import UserInputAction

class Controller(yndf.gui.NethackController):
    """A controller for the YenderFlow GUI debugger."""

    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        self.model = None
        self.obs = None

        self.action_masker = env
        while not hasattr(self.action_masker, 'action_masks'):
            if isinstance(self.action_masker, gym.Wrapper):
                self.action_masker = self.action_masker.env
            else:
                raise ValueError("Environment does not support action masks.")

    def reset(self) -> yndf.NethackState:
        """Reset the controller to the initial state and return the first frame."""
        obs, info = self.env.reset()
        self.obs = obs
        return info["state"]

    def step(self, action: int | None = None) -> yndf.gui.StepInfo:
        """Take a step in the game with the given action, returning StepInfo."""

        if action is None:
            # Predict a maskable action if none is provided
            if self.model is None:
                print("No model set. Please set a model using set_model().")

            action_mask = self.action_masker.action_masks()
            if not any(action_mask):
                raise ValueError("No valid actions available. Check the action mask.")

            action, _ = self.model.predict(self.obs, deterministic=False, action_masks=action_mask)

        else:
            action = UserInputAction(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs

        if (ending := info.get("ending", None)) is not None:
            assert terminated or truncated, "Episode should end if an ending is provided."

        action_mask = info['action_mask']
        available_actions = [ACTIONS[i].name for i, masked in enumerate(action_mask) if masked]
        masked_actions = [ACTIONS[i].name for i, masked in enumerate(action_mask) if not masked]

        state : yndf.NethackState = info["state"]
        properties = {
            "Actions": available_actions,
            "Disallowed": masked_actions,
            "Locked Doors": state.locked_doors,
        }

        if isinstance(action, UserInputAction):
            unwrapped_actions = self.env.unwrapped.actions
            action = unwrapped_actions[unwrapped_actions.index(action.action)].name
        else:
            action = ACTIONS[action].name
        return yndf.gui.StepInfo(state, action, reward,
                                 list(info.get('rewards', {}).items()), properties, ending)

    def set_model(self, model_path: str) -> None:
        """Set the current model path for the controller."""
        self.model = MaskablePPO.load(model_path, env=self.env)

def main():
    """Run the YenderFlow GUI debugger."""
    parser = argparse.ArgumentParser(description="Run the YenderFlow GUI debugger.")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="models/",
        help="Path to a trained model (.zip or base name).",
    )
    args = parser.parse_args()

    env = gym.make("YenderFlow-v0", actions=ACTIONS)
    yndf.gui.run_gui(Controller(env), model_path=args.model_path)

if __name__ == "__main__":
    main()
