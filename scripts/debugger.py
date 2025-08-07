
import gymnasium as gym
import sys
import tty
import termios
from nle import nethack

ACTION_MAP = { x.value: x for x in nethack.ACTIONS }

def read_single_keypress():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # Restore terminal settings

    return ch

PRE_MAP = {'7' : 'y', '8' : 'k', '9' : 'u', '4' : 'h', '6' : 'l', '1' : 'b', '2' : 'j', '3' : 'n'}

def main():
    # Create the NLE environment
    env = gym.make('NetHackChallenge-v0')

    # Reset the environment to start a new episode
    obs, info = env.reset()
    reward = 0.0

    terminated, truncated = False, False
    while not terminated and not truncated:
        print("\033c", end="")
        env.render()
        print("reward:", reward, "score", obs['blstats'][nethack.NLE_BL_SCORE])

        # Read a single keypress from stdin
        c = -1
        while c not in env.unwrapped.actions:
            c = read_single_keypress()
            if c == '\x03':
                print("Exiting...")
                env.close()
                sys.exit(0)

            c = PRE_MAP.get(c, c)  # Map numeric keys to directions
            c = ord(c)

            if c not in env.unwrapped.actions:
                print(f"Invalid action: {chr(c)}")

            else:
                action = env.unwrapped.actions.index(c)

        # Take a step in the environment with the sampled action
        obs, reward, terminated, truncated, info = env.step(action)

    # Close the environment after finishing
    env.close()

if __name__ == "__main__":
    main()
