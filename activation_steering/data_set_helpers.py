from maze_dataset import LatticeMaze, SolvedMaze, MazeDataset, MazeDatasetConfig
from maze_dataset.tokenization import MazeTokenizer, TokenizationMode
import numpy as np

def create_data_sets(dataset, lattice_maze, N):
    variants: list[SolvedMaze] = [
	SolvedMaze(
		connection_list=lattice_maze.connection_list,
		generation_meta=lattice_maze.generation_meta,
		solution=lattice_maze.generate_random_path(),
	)
	for _ in range(N)
    ]
    variants_dataset = MazeDataset(
	    cfg=dataset.cfg,
	    mazes=variants,
    )

    filtered_dataset: MazeDataset = variants_dataset.custom_maze_filter(
	lambda m : len(m.get_coord_neighbors(m.start_pos)) > 1
    )

    return filtered_dataset

def create_data_sets_at_step_s(s, dataset, lattice_maze, N):
    variants: list[SolvedMaze] = [
	SolvedMaze(
		connection_list=lattice_maze.connection_list,
		generation_meta=lattice_maze.generation_meta,
		solution=lattice_maze.generate_random_path(),
	)
	for _ in range(N)
    ]
    variants_dataset = MazeDataset(
	    cfg=dataset.cfg,
	    mazes=variants,
    )

    filtered_dataset: MazeDataset = variants_dataset.custom_maze_filter(
	lambda m : len(m.get_solution_tokens()) > s+2 and len(m.get_coord_neighbors(np.array(m.get_solution_tokens()[s]))) > 1
    )

    return filtered_dataset

def create_directional_data_sets(direction, variants_dataset):

    if direction == 'right' or direction == 'left':
        dataset_right: MazeDataset = variants_dataset.custom_maze_filter(
	    lambda m : m.get_solution_tokens()[1][1] < m.get_solution_tokens()[2][1]
        )

        dataset_left: MazeDataset = variants_dataset.custom_maze_filter(
	    lambda m : m.get_solution_tokens()[1][1] > m.get_solution_tokens()[2][1]
        )
    if direction == 'up' or direction == 'down':
        dataset_up: MazeDataset = variants_dataset.custom_maze_filter(
        lambda m : m.get_solution_tokens()[1][0] > m.get_solution_tokens()[2][0]
        )

        dataset_down: MazeDataset = variants_dataset.custom_maze_filter(
        lambda m : m.get_solution_tokens()[1][0] < m.get_solution_tokens()[2][0]
        )

    if direction == 'right':
        dataset_pos = dataset_right
        dataset_neg = dataset_left
    if direction == 'left':
        dataset_pos = dataset_left
        dataset_neg = dataset_right
    if direction == 'up':
        dataset_pos = dataset_up
        dataset_neg = dataset_down
    if direction == 'down':
        dataset_pos = dataset_down
        dataset_neg = dataset_up

    return dataset_pos, dataset_neg

def create_directional_data_sets_at_step_s(s, direction, variants_dataset):

    if direction == 'right' or direction == 'left':
        dataset_right: MazeDataset = variants_dataset.custom_maze_filter(
	    lambda m : m.get_solution_tokens()[s][1] < m.get_solution_tokens()[s+1][1]
        )

        dataset_left: MazeDataset = variants_dataset.custom_maze_filter(
	    lambda m : m.get_solution_tokens()[s][1] > m.get_solution_tokens()[s+1][1]
        )
    if direction == 'up' or direction == 'down':
        dataset_up: MazeDataset = variants_dataset.custom_maze_filter(
        lambda m : m.get_solution_tokens()[s][0] > m.get_solution_tokens()[s+1][0]
        )

        dataset_down: MazeDataset = variants_dataset.custom_maze_filter(
        lambda m : m.get_solution_tokens()[s][0] < m.get_solution_tokens()[s+1][0]
        )

    if direction == 'right':
        dataset_pos = dataset_right
        dataset_neg = dataset_left
    if direction == 'left':
        dataset_pos = dataset_left
        dataset_neg = dataset_right
    if direction == 'up':
        dataset_pos = dataset_up
        dataset_neg = dataset_down
    if direction == 'down':
        dataset_pos = dataset_down
        dataset_neg = dataset_up

    return dataset_pos, dataset_neg

def create_directional_data_sets_multiple(direction, dataset):

    dataset_neg = []
    dataset_pos = []

    return dataset_pos, dataset_neg