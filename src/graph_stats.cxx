#include <cstdint>
#include <cstdio>
#include <string>

#include <vector>
#include <queue>
#include <stack>

#include <unordered_set>
#include <unordered_map>

#include <iostream>
#include <sstream>
#include <fstream>

#include <algorithm>
#include <chrono>

#include <omp.h>

/// Maximum number of paths to consider for one longer paths
#define MAX_CONSIDERED_NUMBER_OF_OL_PATHS 5
/// Maximum number of paths to consider for two longer paths
#define MAX_CONSIDERED_NUMBER_OF_TL_PATHS 3


// Function to trim whitespace from a string (helper function)
std::string trim(const std::string& str) {
    const auto whitespace = " \t\n\r";
    const size_t start = str.find_first_not_of(whitespace);

    if (start == std::string::npos)
        return ""; // Empty string

    const size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}


std::vector<std::vector<uint16_t>> build_reverse_adj_list(const std::vector<std::vector<uint16_t>>& adj_list) {
    const size_t num_nodes = adj_list.size();
    std::vector<std::vector<uint16_t>> reverse_adj_list(num_nodes);

    for (uint16_t u = 0; u < static_cast<uint16_t>(num_nodes); ++u) {
        for (const uint16_t v : adj_list[u]) {
            reverse_adj_list[v].push_back(u);
        }
    }

    return reverse_adj_list;
}

// Function to reconstruct all shortest paths from source to target
void reconstruct_paths(const uint16_t source, const uint16_t target,
                       const std::vector<std::vector<uint16_t>>& predecessors,
                       std::vector<std::vector<uint16_t>>& paths,
                       std::vector<uint16_t>& current_path) {
    if (target == source) {
        current_path.push_back(source);
        std::vector<uint16_t> path = current_path;
        std::ranges::reverse(path);
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

void reconstruct_paths_of_length(const uint16_t source, const uint16_t target,
                                 const std::vector<std::vector<uint16_t>>& predecessors,
                                 const std::vector<uint16_t>& distances,
                                 const std::vector<std::vector<uint16_t>>& reverse_adj_list,
                                 std::vector<std::vector<uint16_t>>& paths,
                                 std::vector<uint16_t>& current_path,
                                 std::unordered_set<uint16_t>& visited,
                                 const uint16_t desired_length,
                                 const uint16_t max_considered_number_of_paths) {
    if (current_path.size() > desired_length + 1) {
        return;
    }

    if (visited.contains(target)) {
        return;
    }

    if (paths.size() >= max_considered_number_of_paths) {
        return;
    }

    if (target == source) {
        current_path.push_back(source);
        if (current_path.size() == desired_length + 1) {
            std::vector<uint16_t> path = current_path;
            std::ranges::reverse(path);
            paths.push_back(path);
        }
        current_path.pop_back();
        return;
    }

    visited.insert(target);
    current_path.push_back(target);

    const uint16_t current_distance = distances[target];

    if (current_path.size() < desired_length + 1) {
        // Consider predecessors (shortest paths)
        for (const uint16_t pred : predecessors[target]) {
            reconstruct_paths_of_length(source, pred, predecessors, distances, reverse_adj_list, paths, current_path, visited, desired_length, max_considered_number_of_paths);
        }

        // Consider nodes at the same distance (to extend the path by one)
        for (const uint16_t pred : reverse_adj_list[target]) {
            if (distances[pred] == current_distance && pred != target) {
                reconstruct_paths_of_length(source, pred, predecessors, distances, reverse_adj_list, paths, current_path, visited, desired_length, max_considered_number_of_paths);
            }
        }
    }

    current_path.pop_back();
    visited.erase(target);
}


struct PairData {
    /// Shortest path length from source to target
    uint16_t shortest_path_length;
    /// Number of shortest paths from source to target
    uint16_t shortest_path_count;
    /// Maximum degree of node on any shortest path (SP) from source to target
    uint16_t max_sp_node_degree;
    /// Maximum average degree of nodes on any shortest path from source to target
    uint16_t max_sp_average_node_degree;
    /// Average degree off all nodes on all shortest paths from source to target
    uint16_t average_sp_average_node_degree;
    /// Number of paths that are one longer (OL) than the shortest path from source to target
    uint16_t one_longer_path_count;
    /// Maximum degree of node on any path that is OL than the SP from source to target
    uint16_t max_ol_node_degree;
    /// Maximum average degree of nodes on any path that is OL than the SP from source to target
    uint16_t max_ol_average_node_degree;
    /// Average degree off all nodes on all paths that are OL than the SP from source to target
    uint16_t average_ol_average_node_degree;
    /// Number of paths that are two longer (TL) than the shortest path from source to target
    uint16_t two_longer_path_count;
    /// Maximum degree of node on any path that is two longer (TL) than the SP from source to target
    uint16_t max_tl_node_degree;
    /// Maximum average degree of nodes on any path that is TL than the SP from source to target
    uint16_t max_tl_average_node_degree;

    // to_string method
    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "Shortest path length:             " << shortest_path_length << '\n'
            << "Shortest path count:              " << shortest_path_count << '\n'
            << "Max SP node degree:               " << max_sp_node_degree << '\n'
            << "Max SP average node degree:       " << max_sp_average_node_degree << '\n'
            << "Average SP average node degree:   " << average_sp_average_node_degree << '\n'
            << "One longer path count:            " << one_longer_path_count << '\n'
            << "Max OL node degree:               " << max_ol_node_degree << '\n'
            << "Max OL average node degree:       " << max_ol_average_node_degree << '\n'
            << "Average OL average node degree:   " << average_ol_average_node_degree << '\n'
            << "Two longer path count:            " << two_longer_path_count << '\n'
            << "Max TL node degree:               " << max_tl_node_degree << '\n'
            << "Max TL average node degree:       " << max_tl_average_node_degree << '\n';
        return oss.str();
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
                    << pair_data[row][col].max_ol_average_node_degree << ' '
                    << pair_data[row][col].average_ol_average_node_degree << ' '
                    << pair_data[row][col].two_longer_path_count << ' '
                    << pair_data[row][col].max_tl_node_degree << ' '
                    << pair_data[row][col].max_tl_average_node_degree << '\n';
        }
    }

    outfile.close();
}


struct Graph {
private:
    /// Vector to map node IDs to node names (id_to_node[id] = node_name)
    std::vector<std::string> id_to_node;
    /// Mapping from node names to node IDs
    std::unordered_map<std::string, uint16_t> node_to_id;
    /// Adjacency list where adjacency_list[node_id] is a vector of neighbor node IDs
    std::vector<std::vector<uint16_t>> adjacency_list;
    /// Degrees of nodes (number of neighbors)
    std::vector<uint16_t> degrees;

    void compute_node_degrees() {
        size_t num_nodes = this->adjacency_list.size();
        this->degrees.resize(num_nodes);
        for (size_t i = 0; i < num_nodes; ++i) {
            this->degrees[i] = static_cast<uint16_t>(this->adjacency_list[i].size());
        }
    }

    void sort_adjacency_list_according_to_degrees() {
        for (std::vector<uint16_t>& neighbors : this->adjacency_list) {
            std::ranges::sort(neighbors,
                              [this](const uint16_t a, const uint16_t b) {
                                  return this->degrees[a] > this->degrees[b];
                              });
        }
    }

public:
    std::string get_node_name(const uint16_t node_id) const {
        return this->id_to_node[node_id];
    }

    uint16_t get_node_id(const std::string& node_name) const {
        return this->node_to_id.at(node_name);
    }

    size_t get_num_nodes() const {
        return this->adjacency_list.size();
    }

    static Graph load_graph_from_file(const std::string& filename) {
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
            if (line == "# Adjacency Lists") {
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
            // Skip other lines
        }

        infile.close();

        graph.compute_node_degrees();
        // graph.sort_adjacency_list_according_to_degrees();
        return graph;
    }

    void print_graph() const {
        std::cout << "Nodes (" << this->id_to_node.size() << "):\n";
        for (size_t id = 0; id < this->id_to_node.size(); ++id) {
            std::cout << "ID " << id << ": " << this->id_to_node[id] << '\n';
        }
        std::cout << "\nAdjacency Lists:\n";
        for (size_t node_id = 0; node_id < this->adjacency_list.size(); ++node_id) {
            const std::vector<uint16_t>& neighbors = this->adjacency_list[node_id];
            std::cout << "Node " << node_id << " (" << this->id_to_node[node_id] << "): ";
            for (const uint16_t neighbor_id : neighbors) {
                std::cout << neighbor_id << " (" << this->id_to_node[neighbor_id] << ") ";
            }
            std::cout << '\n';
        }
    }

    // Function to perform BFS and record all shortest paths from a given source node
    void bfs_all_shortest_paths(const uint16_t source,
                                std::vector<uint16_t>& distances,
                                std::vector<std::vector<uint16_t>>& predecessors) const {
        const size_t num_nodes = this->adjacency_list.size();
        distances.assign(num_nodes, UINT16_MAX);  // Use UINT16_MAX to represent infinity
        predecessors.assign(num_nodes, std::vector<uint16_t>());

        std::queue<uint16_t> q;
        distances[source] = 0;
        q.push(source);

        while (!q.empty()) {
            uint16_t u = q.front();
            q.pop();

            for (uint16_t v : this->adjacency_list[u]) {
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

    using AllPairsAllShortestPaths = std::vector<std::vector<std::vector<std::vector<uint16_t>>>>;

    AllPairsAllShortestPaths compute_all_shortest_paths() const {
        const uint16_t num_nodes = this->adjacency_list.size();
        AllPairsAllShortestPaths all_paths;
        all_paths.resize(num_nodes, std::vector<std::vector<std::vector<uint16_t>>>(num_nodes));

        std::vector<uint16_t> distances;
        std::vector<std::vector<uint16_t>> predecessors;

        uint32_t completed_nodes = 0;

        for (uint16_t source = 0; source < num_nodes; ++source) {
            this->bfs_all_shortest_paths(source, distances, predecessors);

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

        return all_paths;
    }

    std::tuple<uint16_t, uint16_t, uint16_t> compute_shortest_paths_statistics(const std::vector<std::vector<uint16_t>>& paths) const{
        uint16_t max_path_node_degree = 0;
        uint16_t max_average_path_node_degree = 0;
        uint16_t total_paths_node_degree = 0;

        for (const std::vector<uint16_t> &path: paths) {
            uint16_t path_node_degree = 0;
            uint16_t temp_average_node_degree = 0;

            for (uint16_t node_idx = 0; node_idx < static_cast<uint16_t>(path.size()) - 1; ++node_idx) {
                const uint16_t node = path[node_idx];

                path_node_degree += this->degrees[node];
                total_paths_node_degree += this->degrees[node];
                if (this->degrees[node] > max_path_node_degree) {
                    max_path_node_degree = this->degrees[node];
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

    std::vector<std::vector<PairData>> compute_all_shortest_paths_statistics(
            const bool compute_one_longer_paths = false,
            const bool compute_two_longer_paths = false
    ) const {
        const auto num_nodes = static_cast<uint16_t>(this->adjacency_list.size());

        std::vector<std::vector<PairData>> pair_path_data;
        pair_path_data.resize(num_nodes, std::vector<PairData>(num_nodes));

        const std::vector<std::vector<uint16_t>> reverse_adj_list = build_reverse_adj_list(this->adjacency_list);

        uint32_t completed_nodes = 0;
        const auto& graph = const_cast<Graph&>(*this);

        #pragma omp parallel for default(none) shared(pair_path_data, completed_nodes, reverse_adj_list, graph) \
                                               firstprivate(num_nodes, compute_one_longer_paths, compute_two_longer_paths) \
                                               schedule(runtime)
        for (uint16_t source = 0; source < num_nodes; ++source) {
            std::vector<uint16_t> distances;
            std::vector<std::vector<uint16_t>> predecessors;

            graph.bfs_all_shortest_paths(source, distances, predecessors);

            std::vector<std::vector<uint16_t>> paths;
            std::vector<uint16_t> current_path;
            std::unordered_set<uint16_t> visited;

            for (uint16_t target = 0; target < num_nodes; ++target) {
                if (distances[target] != UINT16_MAX) {
                    paths.clear();
                    current_path.clear();

                    reconstruct_paths(source, target, predecessors, paths, current_path);

                    if (!paths.empty()) {
                        pair_path_data[source][target].shortest_path_length = paths[0].size() - 1;
                        pair_path_data[source][target].shortest_path_count = paths.size();

                        auto sp_statistics = graph.compute_shortest_paths_statistics(paths);
                        pair_path_data[source][target].max_sp_node_degree = std::get<0>(sp_statistics);
                        pair_path_data[source][target].max_sp_average_node_degree = std::get<1>(sp_statistics);
                        pair_path_data[source][target].average_sp_average_node_degree = std::get<2>(sp_statistics);

                        if (compute_one_longer_paths) {
                            paths.clear();
                            current_path.clear();
                            visited.clear();

                            reconstruct_paths_of_length(
                                source, target, predecessors, distances, reverse_adj_list,
                                paths, current_path, visited,
                                pair_path_data[source][target].shortest_path_length + 1,
                                MAX_CONSIDERED_NUMBER_OF_OL_PATHS
                            );

                            if (!paths.empty()) {
                                pair_path_data[source][target].one_longer_path_count = paths.size();

                                auto ol_statistics = graph.compute_shortest_paths_statistics(paths);
                                pair_path_data[source][target].max_ol_node_degree = std::get<0>(ol_statistics);
                                pair_path_data[source][target].max_ol_average_node_degree = std::get<1>(ol_statistics);
                                pair_path_data[source][target].average_ol_average_node_degree = std::get<2>(ol_statistics);
                            }
                        }

                        if (compute_two_longer_paths) {
                            paths.clear();
                            current_path.clear();
                            visited.clear();

                            reconstruct_paths_of_length(source, target, predecessors, distances, reverse_adj_list,
                                                        paths, current_path, visited,
                                                        pair_path_data[source][target].shortest_path_length + 2,
                                                        MAX_CONSIDERED_NUMBER_OF_TL_PATHS);

                            if (!paths.empty()) {
                                pair_path_data[source][target].two_longer_path_count = paths.size();

                                auto ol_statistics = graph.compute_shortest_paths_statistics(paths);
                                pair_path_data[source][target].max_tl_node_degree = std::get<0>(ol_statistics);
                                pair_path_data[source][target].max_tl_average_node_degree = std::get<1>(ol_statistics);
                            }
                        }
                    }
                }
            }

            #pragma omp atomic
            completed_nodes++;

            if (completed_nodes % omp_get_num_threads() == 0 || completed_nodes + omp_get_num_threads() >= num_nodes) {
                printf("Completed %4d nodes out of %d (%4.1f%%)\n", completed_nodes, num_nodes,
                       100.0 * completed_nodes / num_nodes);
            }
        }


        std::cout << "Completed all " << completed_nodes << " nodes\n";

        return pair_path_data;
    }
};


int main() {
    try {
        omp_set_num_threads(16);

        const Graph graph = Graph::load_graph_from_file("../data/paths-and-graph/adj_list.txt");

        // Add timing here
        const auto start = std::chrono::high_resolution_clock::now();
        const std::vector<std::vector<PairData>> pair_path_data = graph.compute_all_shortest_paths_statistics(true, true);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

        dump_to_file("../data/paths-and-graph/pair_stats.txt", pair_path_data);

        constexpr uint16_t max_print_value = 30;

        // Draw some random samples for the pairs and display the results
        for (uint16_t i = 0; i < max_print_value; ++i) {
            const uint16_t source = rand() % graph.get_num_nodes();
            const uint16_t target = rand() % graph.get_num_nodes();

            std::cout << "Shortest path from " << graph.get_node_name(source) << " to " << graph.get_node_name(target)<< ":\n";
            std::cout << pair_path_data[source][target].to_string();
            std::cout << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
