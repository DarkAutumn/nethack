import sys
import gymnasium as gym
from sb3_contrib import MaskablePPO
import yndf

from train import ACTIONS

class Controller(yndf.NethackController):
    """A controller for the YenderFlow GUI debugger."""

    def __init__(self, env: gym.Env, model: MaskablePPO):
        super().__init__()
        self.env = env
        self.model = model
        self.obs = None

    def reset(self) -> yndf.TerminalFrame:
        """Reset the controller to the initial state and return the first frame."""
        obs, info = self.env.reset()
        self.obs = obs
        return self._get_terminal_frame(info["state"])

    def step(self, action: int | None = None) -> yndf.StepInfo:
        """Take a step in the game with the given action, returning StepInfo or EpisodeEnd."""

        if action is None:
            # Predict a maskable action if none is provided
            action, _ = self.model.predict(self.obs, deterministic=False)
        else:
            if action not in ACTIONS:
                print(f"Invalid action: {action}. Must be one of {ACTIONS}.")
                return None

            action = ACTIONS.index(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs

        if (ending := info.get("ending", None)) is not None:
            assert terminated or truncated, "Episode should end if an ending is provided."

        terminal_frame = self._get_terminal_frame(info["state"])
        return yndf.StepInfo(
            frame=terminal_frame,
            action=ACTIONS[action].name,
            reward=reward,
            labels=list(info.get('rewards', {}).items()),
            ending=ending
        )

    def _get_terminal_frame(self, state : yndf.NethackState ) -> yndf.TerminalFrame:
        return yndf.TerminalFrame(chars = state.tty_chars, colors = state.tty_colors)

def main():
    """Run the YenderFlow GUI debugger."""
    env = gym.make("YenderFlow-v0", actions=ACTIONS)

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/ppo_nethack_nav"
    env = gym.make("YenderFlow-v0", actions=ACTIONS)
    model = MaskablePPO.load(model_path, env=env)
    yndf.run_gui(Controller(env, model=model))

if __name__ == "__main__":
    main()
