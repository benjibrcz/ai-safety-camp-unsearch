import numpy as np
from maze_dataset import LatticeMaze, SolvedMaze, MazeDataset, MazeDatasetConfig
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from jaxtyping import Float

def cast_sols_to_arrays(tokenizer, sol):
    # Cast the solutions to arrays
    sol = np.array(tokenizer.strings_to_coords(sol))
    return sol
def direction_to_diff(direction):
    # Convert the direction to a difference
    if direction == 'right':
        return [1,0]
    if direction == 'left':
        return [-1,0]
    if direction == 'up':
        return [0,1]
    if direction == 'down':
        return [0,-1]

def check_correct_steering(perturbed, direction):
    # Check if the steering was successful
    correct_stir = direction_to_diff(direction)
    stirred_diff = perturbed[1] - perturbed[0]

    return all(stirred_diff == correct_stir)

def check_correct_steering_at_step_s(s, perturbed, direction):
    # Check if the steering was successful
    correct_stir = direction_to_diff(direction)
    stirred_diff = perturbed[s+1] - perturbed[s]

    return all(stirred_diff == correct_stir)

def check_correct_steering_batch(originals, perturbeds):
    correct_sum = 0
    for i in range(len(originals)):
        if check_correct_steering(originals[i], perturbeds[i]):
            correct_sum += 1
    return correct_sum/len(originals)

def check_correct_end(original, perturbed):
    # Check if the end of the maze was reached
    return all(perturbed[-1] == original[-1])

def check_errors_along_path(original, perturbed):
    # Check if the perturbed path has errors
    errors = []
    for i in range(len(original[1])):
        if original[1][i] != perturbed[1][i]:
            errors.append(i)
    return errors

def number_of_errors(errors):
    return len(errors)

def get_start_loc(maze):
    start = maze[maze.index("<PATH_START>")+1]

    return tuple(int(x) for x in start.replace("(", "").replace(")", "").split(","))

def plot_directional_heatmap(data: Float[np.ndarray, "n n 4"], cmap: str = "viridis"):
    n: int = data.shape[0]
    assert data.shape == (n, n, 4), f"Expected shape (n, n, 4), got {data.shape}"
    fig, ax = plt.subplots(figsize=(8, 8))

    # Normalizing the data for color mapping
    norm = plt.Normalize(vmin=data.min(), vmax=data.max())
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    kwargs: dict = dict(
        closed=True,
        linewidth=0,
    )

    # Define the corners of each square
    for i in range(n):
        for j in range(n):
            # Coordinates for the center and corners
            center: tuple[float, float] = (j + 0.5, n - i - 0.5)
            top_left: tuple[float, float] = (j, n - i)
            top_right: tuple[float, float] = (j + 1, n - i)
            bottom_left: tuple[float, float] = (j, n - i - 1)
            bottom_right: tuple[float, float] = (j + 1, n - i - 1)

            # Create triangles for each direction
            # Up
            ax.add_patch(patches.Polygon(
                [center, top_left, top_right],
                color=cmap(norm(data[i, j, 0])),
                **kwargs,
            ))
            # Down
            ax.add_patch(patches.Polygon(
                [center, bottom_left, bottom_right],
                color=cmap(norm(data[i, j, 1])),
                **kwargs,
            ))
            # Left
            ax.add_patch(patches.Polygon(
                [center, top_left, bottom_left],
                color=cmap(norm(data[i, j, 2])),
                **kwargs,
            ))
            # Right
            ax.add_patch(patches.Polygon(
                [center, top_right, bottom_right],
                color=cmap(norm(data[i, j, 3])),
                **kwargs,
            ))

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    # ax.axis('off')
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')

    return fig, ax