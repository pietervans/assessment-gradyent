""" Please see readme.md included in the same folder."""

import cProfile
import time
from unittest import TestCase
import unittest

import numpy as np

# ToDo: OPTIONAL - Try to optimize the performance of your solution, enable the unit test by this switch
INCLUDE_OPTIONAL_ASSIGNMENT = True


def graph_traversal_sum(values_in, connections_in, nodes_start, nodes_end):
    """ Calculate the sum of the values in all nodes using the given values in the starting nodes.
    Values in is vectorized, the function is run for m=values_in.shape[0] starting values at once.

    :param values_in: numpy 2d array with initial values of nodes_start of dimension m x n with m=#(different starting values) and n=#nodes
    :param connections_in: numpy 2d array with adjacency matrix of dimension n by n
    :param nodes_start: set with start nodes (leaves of the tree graph)
    :param nodes_end: set with 1 "source node" = root node
    :return values_out: numpy 2d array with the final values for all nodes for all m different starting values, dimension m x n
    """
    # Make sure that original objects are not changed
    values = values_in.copy()
    connections = connections_in.copy()

    # Calculate the graph traversal sum
    assert(len(nodes_end) == 1)
    node_end = next(iter(nodes_end))
    calculate_values_recursive(node_end, values, connections, nodes_start)

    return values


def calculate_values_recursive(node, values, connections_in, nodes_start, node_prev=None) -> None:
    """ Recursive function to calculate the sums for every individual node.

    :param node: the current node
    :param values: numpy 2d array with current values of nodes_start of dimension m x n with m=#(different starting values) and n=#nodes.
        NOTE: the array is passed by reference!
    :param connections_in: numpy 2d array with adjacency matrix of dimension n by n
    :param nodes_start: set with start nodes (leaves of the tree graph)
    :param node_prev: the parent node of the current node
    :return: None
    """

    if node in nodes_start:
        # value is already in the np.ndarray values, we're done!
        pass
    
    else:
        # Now find all child nodes
        neighbours = np.nonzero(connections_in[node])[0]
        neighbours_non_visited = np.array([n for n in neighbours if n != node_prev])

        # Recursive call for all children nodes
        for n in neighbours_non_visited:
            calculate_values_recursive(n, values, connections_in, nodes_start, node_prev = node)
        
        # Calculate values for current node
        values[:,node] = sum([values[:,n] for n in neighbours_non_visited])



class ExampleNetwork1:
    """ Definition of the first example network containing 10 nodes, of which 1 source (9) and 4 sinks (0, 1, 2, 3).

    Schematic overview:

              9
              |
              8
              |
      3 - 6 - 7
              |
              5
              |
              4
            / | \
           0  1  2

    """
    nodes = set(range(0, 10))
    nodes_start = {0, 1, 2, 3}
    nodes_end = {9}
    connections = np.array([
       # 0  1  2  3  4  5  6  7  8  9
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 0
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 1
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 2
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 3
        [1, 1, 1, 0, 0, 1, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 5
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 6
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 9
    ])


class ExampleNetwork2:
    """ A second example network containing 10 nodes, of which 1 source (6) and 4 sinks (0, 1, 4, 9).

    Schematic overview:

              4
              |
              5
              |
          8 - 7 - 9
          |   |
          6   3
              |
              2
            / |
           0  1

    """
    nodes = set(range(0, 10))
    nodes_start = {0, 1, 6, 9}
    nodes_end = {4}
    connections = np.array([
       # 0  1  2  3  4  5  6  7  8  9
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 0
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 1
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 2
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 3
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 4
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 5
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 6
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 1],  # 7
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # 8
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 9
    ])


class TestGraphTraversal(TestCase):

    def test_graph_traversal_sum(self):
        """ Runs the unit test for the graph traversal sum using a given graph traversal method. """
        # Run the test case for the given example network
        values_in1, values_out1, _ = self._run_test_case_for_method(ExampleNetwork1, graph_traversal_sum,
                                                                    number_of_executions=1, length=1000)
        # Input nodes should stay the same
        nodes_start = list(ExampleNetwork1.nodes_start)
        np.testing.assert_array_almost_equal(
            list(values_in1[:, nodes_start]),
            list(values_out1[:, nodes_start]),
            decimal=10,
            err_msg='Network1: Input nodes should stay the same')
        # 'Node 4 should be sum of 0, 1, and 2'
        np.testing.assert_array_almost_equal(
            list(values_out1[:, 4]),
            list(values_out1[:, 0] + values_out1[:, 1] + values_out1[:, 2]),
            decimal=10,
            err_msg='Network1: Node 4 should be sum of 0, 1, and 2')
        # 'Node 6 should be same as 4'
        np.testing.assert_array_almost_equal(
            list(values_out1[:, 6]),
            list(values_out1[:, 3]),
            decimal=10,
            err_msg='Network1: Node 6 should be same as 3')
        # 'Node 9 should be sum of 3 and 4'
        np.testing.assert_array_almost_equal(
            list(values_out1[:, 9]),
            list((values_out1[:, 3] + values_out1[:, 4])),
            decimal=10,
            err_msg='Network1: Node 9 should be sum of 3 and 4')

        # Run the test case for the second example network
        values_in2, values_out2, _ = self._run_test_case_for_method(ExampleNetwork2, graph_traversal_sum,
                                                                    number_of_executions=1, length=1000)
        # Input nodes should stay the same
        nodes_start = list(ExampleNetwork2.nodes_start)
        np.testing.assert_array_almost_equal(
            list(values_in2[:, nodes_start]),
            list(values_out2[:, nodes_start]),
            decimal=10,
            err_msg='Network2: Input nodes should stay the same')
        # 'Node 7 should be sum of 8, 3, 9'
        np.testing.assert_array_almost_equal(
            list(values_out2[:, 7]),
            list(values_out2[:, 8] + values_out2[:, 3] + values_out2[:, 9]),
            decimal=10,
            err_msg='Network2: Node 7 should be sum of 5, 3, 9')
        # 'Node 6 should be same as 8'
        np.testing.assert_array_almost_equal(
            list(values_out2[:, 6]),
            list(values_out2[:, 8]),
            decimal=10,
            err_msg='Network2: Node 6 should be same as 8')
        # 'Node 4 should be sum of 6, 0, 1, 9'
        np.testing.assert_array_almost_equal(
            list(values_out2[:, 4]),
            list((values_out2[:, 6] + values_out2[:, 9] + values_out2[:, 0] + values_out2[:, 1])),
            decimal=10,
            err_msg='Network2: Node 6 should be sum of 4, 0, 1, 9')

    def test_graph_traversal_sum_performance(self):
        """ Evaluates the performance of a given graph traversal method if the optional assignment is included. """
        if INCLUDE_OPTIONAL_ASSIGNMENT:
            # The goal in seconds of the average runtime
            average_run_time_goal = 0.001

            # Start the profiler
            pr = cProfile.Profile()
            pr.enable()

            # Run the test case for the given example network (a large number of times)
            values_in, values_out, average_run_time = self._run_test_case_for_method(
                ExampleNetwork1, graph_traversal_sum, number_of_executions=1000, length=10000)

            # Stop and print the profiling information
            pr.disable()
            pr.print_stats(sort="calls")

            # Try to get the run time below this value
            self.assertLess(average_run_time, average_run_time_goal,
                            msg=f'OPTIONAL: Try to get the average runtime to below {average_run_time_goal}.')
        else:
            # Do not test anything when the optional assignment is not included
            self.assertFalse(INCLUDE_OPTIONAL_ASSIGNMENT,
                             msg='The optional assignment is turned off, '
                                 'put "INCLUDE_OPTIONAL_ASSIGNMENT" to True to include it.')

    @staticmethod
    def _run_test_case_for_method(network_to_use, method_to_use, number_of_executions=1, length=10000):
        # Initialize the values for all nodes and the given length
        values = np.zeros((length, len(network_to_use.nodes)))
        # Create random values for all starting nodes
        values[:, list(network_to_use.nodes_start)] = np.random.random((length, len(network_to_use.nodes_start)))

        # Run an x-amount of times and note down the total execution time
        # NOTE: This is only relevant for the optional assignment, where you are optimizing the performance
        start = time.time()
        values_out = values.copy()
        for i in range(0, number_of_executions):
            values_out = method_to_use(values, network_to_use.connections,
                                       network_to_use.nodes_start,  network_to_use.nodes_end)
        end = time.time()
        run_time = end - start
        average_run_time = run_time / number_of_executions
        print(f'Execution time for {number_of_executions} run(s): {run_time}')

        return values, values_out, average_run_time


if __name__ == '__main__':
    unittest.main()
