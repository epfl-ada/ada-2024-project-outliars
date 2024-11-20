#include <cstdint>
#include <vector>
#include <unordered_map>

#include <iostream>
#include <sstream>
#include <fstream>

#include <algorithm>
#include <queue>
#include <unordered_set>

#include <omp.h>


struct Graph {
    /// Vector to map node IDs to node names (id_to_node[id] = node_name)
    std::vector<std::string> id_to_node;
    /// Mapping from node names to node IDs
    std::unordered_map<std::string, uint16_t> node_to_id;
    /// Adjacency list where adjacency_list[node_id] is a vector of neighbor node IDs
    std::vector<std::vector<uint16_t>> adjacency_list;
    /// Degrees of nodes (number of neighbors)
    std::vector<uint16_t> degrees;


};

struct PairData {
    /// Shortest path length from source to target
    uint16_t shortest_path_length;
    /// Number of shortest paths from source to target
    uint16_t shortest_path_count;
    /// Maximum degree of node on any shortest path (SP) from source to target
    uint16_t max_sp_node_degree;
    /// Maximum average degree of nodes on any shortest path from source to target
    uint16_t max_sp_average_node_degree;
    /// average degree off all nodes on all shortest paths from source to target
    uint16_t average_sp_average_node_degree;
    /// Number of paths that are one longer (OL) than the shortest path from source to target
    uint16_t one_longer_path_count;
    /// Maximum degree of node on any path that is OL than the SP from source to target
    uint16_t max_ol_node_degree;
    /// Maximum average degree of nodes on any path that is OL than the SP from source to target
    uint16_t max_ol_average_node_degree;

    void print() const {
        std::cout << "Shortest path length:             " << shortest_path_length << '\n';
        std::cout << "Shortest path count:              " << shortest_path_count << '\n';
        std::cout << "Max SP node degree:               " << max_sp_node_degree << '\n';
        std::cout << "Max SP average node degree:       " << max_sp_average_node_degree << '\n';
        std::cout << "Average SP average node degree:   " << average_sp_average_node_degree << '\n';
        std::cout << "One longer path count:            " << one_longer_path_count << '\n';
        std::cout << "Max OL node degree:               " << max_ol_node_degree << '\n';
        std::cout << "Max OL average node degree:       " << max_ol_average_node_degree << '\n';
    }
};

void dump_to_file(const std::string& filename, const std::vector<std::vector<PairData>>& pair_data) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    for (uint16_t row = 0; row < static_cast<uint16_t>(pair_data.size()); ++row) {
        for (uint16_t col = 0; col < static_cast<uint16_t>(pair_data[row].size()); ++col) {
            if (pair_data[row][col].shortest_path_count == 0) {
                continue;
            }

            outfile << row << ' ' << col << ' '
                    << pair_data[row][col].shortest_path_length << ' '
                    << pair_data[row][col].shortest_path_count << ' '
                    << pair_data[row][col].max_sp_node_degree << ' '
                    << pair_data[row][col].max_sp_average_node_degree << ' '
                    << pair_data[row][col].average_sp_average_node_degree << ' '
                    << pair_data[row][col].one_longer_path_count << ' '
                    << pair_data[row][col].max_ol_node_degree << ' '
                    << pair_data[row][col].max_ol_average_node_degree << '\n';
        }
    }

    outfile.close();
}

// Function to compute degrees of all nodes
void compute_node_degrees(Graph& graph) {
    size_t num_nodes = graph.adjacency_list.size();
    graph.degrees.resize(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        graph.degrees[i] = static_cast<uint16_t>(graph.adjacency_list[i].size());
    }
}

// Function to trim whitespace from a string (helper function)
std::string trim(const std::string& str) {
    const char* whitespace = " \t\n\r";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos)
        return ""; // Empty string
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

// Function to load the graph from the file
Graph load_graph_from_file(const std::string& filename) {
    Graph graph;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::string line;
    bool reading_index = false;
    bool reading_adjacency = false;

    // Temporary storage for node IDs and names
    std::vector<std::pair<uint16_t, std::string>> id_name_pairs;

    // Read the file line by line
    while (std::getline(infile, line)) {
        line = trim(line);

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        // Check for headers
        if (line == "# Node Index Mapping") {
            reading_index = true;
            reading_adjacency = false;
            continue;
        }
        else if (line == "# Adjacency Lists") {
            reading_index = false;
            reading_adjacency = true;

            // Initialize the id_to_node vector based on collected IDs
            size_t num_nodes = id_name_pairs.size();
            graph.id_to_node.resize(num_nodes);
            graph.adjacency_list.resize(num_nodes);

            // Populate id_to_node vector and node_to_id map
            for (const auto&[fst, snd] : id_name_pairs) {
                uint16_t id = fst;
                const std::string& node_name = snd;
                graph.id_to_node[id] = node_name;
                graph.node_to_id[node_name] = id;
            }

            continue;
        }

        if (reading_index) {
            // Parse index mapping lines: "<ID>\t<node_name>"
            std::istringstream iss(line);
            uint16_t id;
            std::string node_name;
            if (!(iss >> id >> std::ws) || !std::getline(iss, node_name)) {
                throw std::runtime_error("Error parsing index mapping line: " + line);
            }
            node_name = trim(node_name);

            // Store the id and node name in the temporary vector
            id_name_pairs.emplace_back(id, node_name);
        }
        else if (reading_adjacency) {
            // Parse adjacency list lines: "<node_ID>: <neighbor_ID1> <neighbor_ID2> ..."
            size_t colon_pos = line.find(':');
            if (colon_pos == std::string::npos) {
                throw std::runtime_error("Error parsing adjacency list line (missing ':'): " + line);
            }

            // Extract node ID
            std::string node_id_str = trim(line.substr(0, colon_pos));
            uint16_t node_id = std::stoul(node_id_str);

            // Extract neighbor IDs
            std::string neighbors_str = trim(line.substr(colon_pos + 1));
            std::istringstream neighbors_stream(neighbors_str);

            std::string neighbor_id_str;
            std::vector<uint16_t> neighbors;

            while (neighbors_stream >> neighbor_id_str) {
                uint16_t neighbor_id = std::stoul(neighbor_id_str);
                neighbors.push_back(neighbor_id);
            }

            // Assign the neighbors vector to the adjacency_list at node_id
            graph.adjacency_list[node_id] = std::move(neighbors);
        }
        else {
            // Skip any lines outside the expected sections
            continue;
        }
    }

    infile.close();

    compute_node_degrees(graph);
    return graph;
}

std::vector<std::vector<uint16_t>> build_reverse_adj_list(const std::vector<std::vector<uint16_t>>& adj_list) {
    size_t num_nodes = adj_list.size();
    std::vector<std::vector<uint16_t>> reverse_adj_list(num_nodes);

    for (uint16_t u = 0; u < num_nodes; ++u) {
        for (uint16_t v : adj_list[u]) {
            reverse_adj_list[v].push_back(u);
        }
    }

    return reverse_adj_list;
}

// Function to print the graph (for testing purposes)
void print_graph(const Graph& graph) {
    std::cout << "Nodes (" << graph.id_to_node.size() << "):\n";
    for (size_t id = 0; id < graph.id_to_node.size(); ++id) {
        std::cout << "ID " << id << ": " << graph.id_to_node[id] << '\n';
    }
    std::cout << "\nAdjacency Lists:\n";
    for (size_t node_id = 0; node_id < graph.adjacency_list.size(); ++node_id) {
        const std::vector<uint16_t>& neighbors = graph.adjacency_list[node_id];
        std::cout << "Node " << node_id << " (" << graph.id_to_node[node_id] << "): ";
        for (uint16_t neighbor_id : neighbors) {
            std::cout << neighbor_id << " (" << graph.id_to_node[neighbor_id] << ") ";
        }
        std::cout << '\n';
    }
}

// Function to perform BFS and record all shortest paths from a given source node
void bfs_all_shortest_paths(const Graph& graph, uint16_t source,
                            std::vector<uint16_t>& distances,
                            std::vector<std::vector<uint16_t>>& predecessors) {
    size_t num_nodes = graph.adjacency_list.size();
    distances.assign(num_nodes, UINT16_MAX);  // Use UINT16_MAX to represent infinity
    predecessors.assign(num_nodes, std::vector<uint16_t>());

    std::queue<uint16_t> q;
    distances[source] = 0;
    q.push(source);

    while (!q.empty()) {
        uint16_t u = q.front();
        q.pop();

        for (uint16_t v : graph.adjacency_list[u]) {
            // If v is found for the first time
            if (distances[v] == UINT16_MAX) {
                distances[v] = distances[u] + 1;
                predecessors[v].push_back(u);
                q.push(v);
            }
                // If v is found again at the same level, it's another shortest path
            else if (distances[v] == distances[u] + 1) {
                predecessors[v].push_back(u);
            }
        }
    }
}

// Function to reconstruct all shortest paths from source to target
void reconstruct_paths(uint16_t source, uint16_t target,
                       const std::vector<std::vector<uint16_t>>& predecessors,
                       std::vector<std::vector<uint16_t>>& paths,
                       std::vector<uint16_t>& current_path) {
    if (target == source) {
        current_path.push_back(source);
        std::vector<uint16_t> path = current_path;
        std::reverse(path.begin(), path.end());
        paths.push_back(path);
        current_path.pop_back();
        return;
    }
    current_path.push_back(target);
    for (uint16_t pred : predecessors[target]) {
        reconstruct_paths(source, pred, predecessors, paths, current_path);
    }
    current_path.pop_back();
}

void reconstruct_paths_of_length(uint16_t source, uint16_t target,
                                 const std::vector<std::vector<uint16_t>>& predecessors,
                                 const std::vector<uint16_t>& distances,
                                 const std::vector<std::vector<uint16_t>>& reverse_adj_list,
                                 std::vector<std::vector<uint16_t>>& paths,
                                 std::vector<uint16_t>& current_path,
                                 std::unordered_set<uint16_t>& visited,
                                 uint16_t desired_length,
                                 uint16_t max_considered_number_of_paths) {
    if (current_path.size() > desired_length + 1) {
        return;
    }

    if (visited.count(target)) {
        return;
    }

    if (paths.size() >= max_considered_number_of_paths) {
        return;
    }

    if (target == source) {
        current_path.push_back(source);
        if (current_path.size() == desired_length + 1) {
            std::vector<uint16_t> path = current_path;
            std::reverse(path.begin(), path.end());
            paths.push_back(path);
        }
        current_path.pop_back();
        return;
    }

    visited.insert(target);
    current_path.push_back(target);

    uint16_t current_distance = distances[target];

    if (current_path.size() < desired_length + 1) {
        // Consider predecessors (shortest paths)
        for (uint16_t pred : predecessors[target]) {
            reconstruct_paths_of_length(source, pred, predecessors, distances, reverse_adj_list, paths, current_path, visited, desired_length, max_considered_number_of_paths);
        }

        // Consider nodes at the same distance (to extend the path by one)
        for (uint16_t pred : reverse_adj_list[target]) {
            if (distances[pred] == current_distance && pred != target) {
                reconstruct_paths_of_length(source, pred, predecessors, distances, reverse_adj_list, paths, current_path, visited, desired_length, max_considered_number_of_paths);
            }
        }
    }

    current_path.pop_back();
    visited.erase(target);
}

// Function to compute all shortest paths between all pairs
void compute_all_shortest_paths(const Graph& graph,
                                std::vector<std::vector<std::vector<std::vector<uint16_t>>>>& all_paths) {
    uint16_t num_nodes = graph.adjacency_list.size();
    all_paths.resize(num_nodes, std::vector<std::vector<std::vector<uint16_t>>>(num_nodes));

    std::vector<uint16_t> distances;
    std::vector<std::vector<uint16_t>> predecessors;

    uint32_t completed_nodes = 0;

    for (uint16_t source = 0; source < num_nodes; ++source) {
        bfs_all_shortest_paths(graph, source, distances, predecessors);

        // For each target node, reconstruct all shortest paths from source to target
        for (uint16_t target = 0; target < num_nodes; ++target) {
            if (distances[target] != UINT16_MAX) {
                std::vector<std::vector<uint16_t>> paths;
                std::vector<uint16_t> current_path;
                reconstruct_paths(source, target, predecessors, paths, current_path);
                all_paths[source][target] = std::move(paths);
            }
            // If there is no path, the vector remains empty
        }
        completed_nodes++;
        std::cout << "Completed " << completed_nodes << " nodes out of " << num_nodes << '\n';
    }
}

std::tuple<uint16_t, uint16_t, uint16_t> compute_shortest_paths_statistics(const Graph& graph,
                                                                          const std::vector<std::vector<uint16_t>>& paths) {
    uint16_t max_path_node_degree = 0;
    uint16_t max_average_path_node_degree = 0;
    uint16_t total_paths_node_degree = 0;

    for (const std::vector<uint16_t> &path: paths) {
        uint16_t path_node_degree = 0;
        uint16_t temp_average_node_degree = 0;

        for (uint16_t node_idx = 0; node_idx < static_cast<uint16_t>(path.size()) - 1; ++node_idx) {
            const uint16_t node = path[node_idx];

            path_node_degree += graph.degrees[node];
            total_paths_node_degree += graph.degrees[node];
            if (graph.degrees[node] > max_path_node_degree) {
                max_path_node_degree = graph.degrees[node];
            }
        }

        temp_average_node_degree = path_node_degree / path.size();

        if (temp_average_node_degree > max_average_path_node_degree) {
            max_average_path_node_degree = temp_average_node_degree;
        }
    }

    total_paths_node_degree /= paths.size() * paths[0].size();

    return {max_path_node_degree, max_average_path_node_degree, total_paths_node_degree};
}

#define MAX_CONSIDERED_NUMBER_OF_PATHS 5

void compute_all_shortest_paths_statistics(const Graph& graph,
                                           std::vector<std::vector<PairData>>& pair_path_data,
                                           const bool compute_one_longer_paths = false) {
    uint16_t num_nodes = graph.adjacency_list.size();
    pair_path_data.resize(num_nodes, std::vector<PairData>(num_nodes));

    std::vector<std::vector<uint16_t>> reverse_adj_list = build_reverse_adj_list(graph.adjacency_list);
    uint32_t completed_nodes = 0;

#pragma omp parallel default(shared)
    {
        std::vector<uint16_t> distances;
        std::vector<std::vector<uint16_t>> predecessors;

        #pragma omp for schedule(dynamic)
        for (uint16_t source = 0; source < static_cast<uint16_t>(graph.adjacency_list.size()); ++source) {
            bfs_all_shortest_paths(graph, source, distances, predecessors);

            std::vector<std::vector<uint16_t>> paths;
            std::vector<uint16_t> current_path;
            std::unordered_set<uint16_t> visited;

            for (uint16_t target = 0; target < static_cast<uint16_t>(graph.adjacency_list.size()); ++target) {
                if (distances[target] != UINT16_MAX) {
                    paths.clear();
                    current_path.clear();

                    reconstruct_paths(source, target, predecessors, paths, current_path);

                    if (!paths.empty()) {
                        pair_path_data[source][target].shortest_path_length = paths[0].size() - 1;
                        pair_path_data[source][target].shortest_path_count = paths.size();

                        auto sp_statistics = compute_shortest_paths_statistics(graph, paths);
                        pair_path_data[source][target].max_sp_node_degree = std::get<0>(sp_statistics);
                        pair_path_data[source][target].max_sp_average_node_degree = std::get<1>(sp_statistics);
                        pair_path_data[source][target].average_sp_average_node_degree = std::get<2>(sp_statistics);

                        if (compute_one_longer_paths) {
                            paths.clear();
                            current_path.clear();
                            visited.clear();

                            reconstruct_paths_of_length(source, target, predecessors, distances, reverse_adj_list,
                                                        paths, current_path, visited, paths[0].size(), MAX_CONSIDERED_NUMBER_OF_PATHS);

                            if (!paths.empty()) {
                                pair_path_data[source][target].one_longer_path_count = paths.size();

                                auto ol_statistics = compute_shortest_paths_statistics(graph, paths);
                                pair_path_data[source][target].max_ol_node_degree = std::get<0>(ol_statistics);
                                pair_path_data[source][target].max_ol_average_node_degree = std::get<1>(ol_statistics);
                            }
                        }
                    }
                }
            }
            #pragma omp critical
            {
                completed_nodes++;
                if (completed_nodes % 10 == 0) {
                    printf("Completed %d nodes out of %d\n", completed_nodes, num_nodes);
                }
            }
        }
    }

    std::cout << "Completed " << completed_nodes << " nodes\n";
}


int main() {
    try {
        omp_set_num_threads(16);

        Graph graph = load_graph_from_file("../data/paths-and-graph/adj_list.txt");

        std::vector<std::vector<PairData>> pair_path_data;

        compute_all_shortest_paths_statistics(graph, pair_path_data, true);

        dump_to_file("../data/paths-and-graph/pair_stats.txt", pair_path_data);

        const uint16_t max_print_value = 30;

        // Draw some random samples for the pairs and display the results
        for (uint16_t i = 0; i < max_print_value; ++i) {
            uint16_t source = rand() % graph.adjacency_list.size();
            uint16_t target = rand() % graph.adjacency_list.size();

            std::cout << "Shortest path from " << graph.id_to_node[source] << " to " << graph.id_to_node[target] << ":\n";
            pair_path_data[source][target].print();
            std::cout << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
