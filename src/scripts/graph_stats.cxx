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
#define MAX_CONSIDERED_NUMBER_OF_OL_PATHS 30000
/// Maximum number of paths to consider for two longer paths
#define MAX_CONSIDERED_NUMBER_OF_TL_PATHS 300


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
    if (current_path.size() > static_cast<uint16_t>(desired_length + 1)) {
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
        if (current_path.size() == static_cast<uint16_t>(desired_length + 1)) {
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

    if (current_path.size() < static_cast<uint16_t>(desired_length + 1)) {
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


struct PairDataPagerank {
    /// Source node ID
    uint16_t source;
    /// Target node ID
    uint16_t target;
    /// Shortest path length from source to target
    uint16_t shortest_path_length;
    /// Number of shortest paths from source to target
    uint16_t shortest_path_count;
    /// Maximum pagerank of node on any shortest path (SP) from source to target
    float max_sp_pagerank;
    /// Maximum average pagerank of nodes on any shortest path from source to target
    float max_sp_average_pagerank;
    /// Average pagerank off all nodes on all shortest paths from source to target
    float average_sp_average_pagerank;
    /// Number of paths that are one longer (OL) than the shortest path from source to target
    uint32_t one_longer_path_count;
    /// Maximum pagerank of node on any path that is OL than the SP from source to target
    float max_ol_pagerank;
    /// Maximum average pagerank of nodes on any path that is OL than the SP from source to target
    float max_ol_average_pagerank;
    /// Average pagerank off all nodes on all paths that are OL than the SP from source to target
    float average_ol_average_pagerank;
    /// Number of paths that are two longer (TL) than the shortest path from source to target
    uint32_t two_longer_path_count;
    /// Maximum pagerank of node on any path that is two longer (TL) than the SP from source to target
    float max_tl_pagerank;
    /// Maximum average pagerank of nodes on any path that is TL than the SP from source to target
    float max_tl_average_pagerank;
    // /// Average pagerank off all nodes on all paths that are TL than the SP from source to target
    // float average_tl_average_pagerank;

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "Source:                         " << source << '\n'
            << "Target:                         " << target << '\n'
            << "Shortest path length:           " << shortest_path_length << '\n'
            << "Shortest path count:            " << shortest_path_count << '\n'
            << "Max SP pagerank:                " << max_sp_pagerank << '\n'
            << "Max SP average pagerank:        " << max_sp_average_pagerank << '\n'
            << "Average SP average pagerank:    " << average_sp_average_pagerank << '\n'
            << "One longer path count:          " << one_longer_path_count << '\n'
            << "Max OL pagerank:                " << max_ol_pagerank << '\n'
            << "Max OL average pagerank:        " << max_ol_average_pagerank << '\n'
            << "Average OL average pagerank:    " << average_ol_average_pagerank << '\n'
            << "Two longer path count:          " << two_longer_path_count << '\n'
            << "Max TL pagerank:                " << max_tl_pagerank << '\n'
            << "Max TL average pagerank:        " << max_tl_average_pagerank << '\n';
            // << "Average TL average pagerank:    " << average_tl_average_pagerank << '\n';
        return oss.str();
    }
};


struct NodeData {
    /// Node name
    std::string name;
    /// Node degree
    uint16_t degree{};
    /// Node closeness centrality
    double closeness_centrality{};
    /// Node betweenness centrality
    double betweenness_centrality{};
    /// Node PageRank
    double pagerank{};

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "Node name:               " << name << '\n'
            << "Node degree:             " << degree << '\n'
            << "Closeness centrality:    " << closeness_centrality << '\n'
            << "Betweenness centrality:  " << betweenness_centrality << '\n'
            << "PageRank:                " << pagerank << '\n';
        return oss.str();
    }
};

void dump_node_data_to_file(const std::string& filename, const std::vector<NodeData>& node_data) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    // Use scientific notation for floating point numbers and keep 4 decimal places
    for (uint16_t i = 0; i < static_cast<uint16_t>(node_data.size()); ++i) {
        outfile << node_data[i].name << '\t'
                << node_data[i].degree << '\t'
                << node_data[i].closeness_centrality << '\t'
                << node_data[i].betweenness_centrality << '\t'
                << node_data[i].pagerank << '\n';
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
    /// PageRank values of nodes
    std::vector<double> pageranks;

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

    void compute_pageranks() {
        this->pageranks = this->compute_pagerank();
    }

public:
    std::string get_node_name(const uint16_t node_id) const {
        return this->id_to_node[node_id];
    }

    uint16_t get_node_id(const std::string& node_name) const {
        return this->node_to_id.at(node_name);
    }

    uint16_t get_node_degree(const uint16_t node_id) const {
        return this->degrees[node_id];
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
        graph.compute_pageranks();
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

    std::vector<std::pair<uint16_t, uint16_t>> load_pairs_from_file(const std::string& filename) const {
        std::vector<std::pair<uint16_t, uint16_t>> pairs;
        std::ifstream infile(filename);

        if (!infile.is_open()) {
            throw std::runtime_error("Error opening file: " + filename);
        }

        // The line contains tab separated node names
        std::string line;
        while (std::getline(infile, line)) {
            line = trim(line);
            if (line.empty()) {
                continue;
            }

            std::istringstream iss(line);
            std::string node_name1, node_name2;
            if (!(iss >> node_name1 >> node_name2)) {
                throw std::runtime_error("Error parsing pair line: " + line);
            }

            bool found = true;
            // Check whether the name exists in the map
            if (!this->node_to_id.contains(node_name1)) {
                std::cerr << "Node name not found in the graph: " << node_name1 << '\n';
                found = false;
            }
            if (!this->node_to_id.contains(node_name2)) {
                std::cerr << "Node name not found in the graph: " << node_name2 << '\n';
                found = false;
            }

            if (found) {
                uint16_t node_id1 = this->get_node_id(node_name1);
                uint16_t node_id2 = this->get_node_id(node_name2);
                pairs.emplace_back(node_id1, node_id2);
            }
        }

        return pairs;
    }

    void dump_pair_stats_to_file(const std::string& filename, const std::vector<PairDataPagerank>& pair_data) const {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            throw std::runtime_error("Error opening file: " + filename);
        }

        for (const auto & pair : pair_data) {
            std::string source_name = this->get_node_name(pair.source);
            std::string target_name = this->get_node_name(pair.target);

            outfile << source_name << '\t'
                    << target_name << '\t'
                    << pair.shortest_path_length << '\t'
                    << pair.shortest_path_count << '\t'
                    << pair.max_sp_pagerank << '\t'
                    << pair.max_sp_average_pagerank << '\t'
                    << pair.average_sp_average_pagerank << '\t'
                    << pair.one_longer_path_count << '\t'
                    << pair.max_ol_pagerank << '\t'
                    << pair.max_ol_average_pagerank << '\t'
                    << pair.average_ol_average_pagerank << '\t'
                    << pair.two_longer_path_count << '\t'
                    << pair.max_tl_pagerank << '\t'
                    << pair.max_tl_average_pagerank << '\n';
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

    std::tuple<float, float, float> compute_paths_statistics(const std::vector<std::vector<uint16_t>>& paths) const{
        float max_path_pagerank = 0;
        float max_average_path_pagerank = 0;
        float total_paths_pagerank = 0;

        for (const std::vector<uint16_t> &path: paths) {
            float path_pagerank = 0;
            float temp_average_pagerank = 0;

            for (uint16_t node_idx = 0; node_idx < static_cast<uint16_t>(path.size()) - 1; ++node_idx) {
                const uint16_t node = path[node_idx];
                const auto node_pagerank = static_cast<float>(this->pageranks[node]);

                path_pagerank += node_pagerank;
                total_paths_pagerank += node_pagerank;
                if (this->pageranks[node] > max_path_pagerank) {
                    max_path_pagerank = node_pagerank;
                }
            }

            temp_average_pagerank = path_pagerank / static_cast<float>(path.size() - 1);

            if (temp_average_pagerank > max_average_path_pagerank) {
                max_average_path_pagerank = temp_average_pagerank;
            }
        }

        total_paths_pagerank /= static_cast<float>(paths.size() * (paths[0].size() - 1));

        return {max_path_pagerank, max_average_path_pagerank, total_paths_pagerank};
    }

    std::vector<PairDataPagerank> compute_statistics_for_pairs(
            const std::vector<std::pair<uint16_t, uint16_t>>& pairs,
            const bool compute_one_longer_paths = false,
            const bool compute_two_longer_paths = false
    ) const {
        const auto num_pairs = static_cast<uint32_t>(pairs.size());

        std::vector<PairDataPagerank> pair_path_data(num_pairs);

        uint32_t completed_nodes = 0;
        const auto& graph = const_cast<Graph&>(*this);

        const std::vector<std::vector<uint16_t>> reverse_adj_list = build_reverse_adj_list(this->adjacency_list);

        #pragma omp parallel for default(none) shared(pair_path_data, completed_nodes, reverse_adj_list, graph, pairs) \
                                               firstprivate(num_pairs, compute_one_longer_paths, compute_two_longer_paths) \
                                               schedule(dynamic)
        for (size_t pair_idx = 0; pair_idx < num_pairs; ++pair_idx) {
            const uint16_t source = pairs[pair_idx].first;
            const uint16_t target = pairs[pair_idx].second;

            pair_path_data[pair_idx].source = source;
            pair_path_data[pair_idx].target = target;

            std::vector<uint16_t> distances;
            std::vector<std::vector<uint16_t>> predecessors;

            graph.bfs_all_shortest_paths(source, distances, predecessors);

            if (distances[target] != UINT16_MAX) {

                std::vector<std::vector<uint16_t>> paths;
                std::vector<uint16_t> current_path;

                paths.clear();
                current_path.clear();

                reconstruct_paths(source, target, predecessors, paths, current_path);

                if (!paths.empty()) {
                    pair_path_data[pair_idx].shortest_path_length = paths[0].size() - 1;
                    pair_path_data[pair_idx].shortest_path_count = paths.size();

                    auto sp_statistics = graph.compute_paths_statistics(paths);
                    pair_path_data[pair_idx].max_sp_pagerank = std::get<0>(sp_statistics);
                    pair_path_data[pair_idx].max_sp_average_pagerank = std::get<1>(sp_statistics);
                    pair_path_data[pair_idx].average_sp_average_pagerank = std::get<2>(sp_statistics);

                    std::unordered_set<uint16_t> visited;

                    if (compute_one_longer_paths) {
                        paths.clear();
                        current_path.clear();

                        reconstruct_paths_of_length(
                            source, target, predecessors, distances, reverse_adj_list,
                            paths, current_path, visited,
                            pair_path_data[pair_idx].shortest_path_length + 1,
                            MAX_CONSIDERED_NUMBER_OF_OL_PATHS
                        );

                        if (!paths.empty()) {
                            pair_path_data[pair_idx].one_longer_path_count = paths.size();

                            auto ol_statistics = graph.compute_paths_statistics(paths);
                            pair_path_data[pair_idx].max_ol_pagerank = std::get<0>(ol_statistics);
                            pair_path_data[pair_idx].max_ol_average_pagerank= std::get<1>(ol_statistics);
                            pair_path_data[pair_idx].average_ol_average_pagerank = std::get<2>(ol_statistics);
                        }
                    }

                    if (compute_two_longer_paths) {
                        paths.clear();
                        current_path.clear();
                        visited.clear();

                        reconstruct_paths_of_length(source, target, predecessors, distances, reverse_adj_list,
                                                    paths, current_path, visited,
                                                    pair_path_data[pair_idx].shortest_path_length + 2,
                                                    MAX_CONSIDERED_NUMBER_OF_TL_PATHS);

                        if (!paths.empty()) {
                            pair_path_data[pair_idx].two_longer_path_count = paths.size();

                            auto ol_statistics = graph.compute_paths_statistics(paths);
                            pair_path_data[pair_idx].max_tl_pagerank = std::get<0>(ol_statistics);
                            pair_path_data[pair_idx].max_tl_average_pagerank = std::get<1>(ol_statistics);
                        }
                    }
                }
            }

            #pragma omp atomic
            completed_nodes++;

            if (completed_nodes % (20 * omp_get_num_threads()) == 0 || completed_nodes + omp_get_num_threads() >= num_pairs) {
                printf("Completed %4d nodes out of %d (%4.1f%%)\n", completed_nodes, num_pairs,
                       100.0 * completed_nodes / num_pairs);
            }
        }


        std::cout << "Completed all " << completed_nodes << " nodes\n";

        return pair_path_data;
    }

    std::vector<double> compute_closeness_centrality() const {
        const size_t num_nodes = this->adjacency_list.size();
        std::vector<double> closeness_centrality(num_nodes, 0.0);

        #pragma omp parallel for default(shared) schedule(dynamic)
        for (uint16_t source = 0; source < static_cast<uint16_t>(num_nodes); ++source) {
            std::vector<uint16_t> distances(num_nodes, UINT16_MAX);
            std::queue<uint16_t> q;

            distances[source] = 0;
            q.push(source);

            uint32_t sum_distances = 0;
            uint32_t reachable_nodes = 0;

            while (!q.empty()) {
                uint16_t u = q.front();
                q.pop();

                for (uint16_t v : this->adjacency_list[u]) {
                    if (distances[v] == UINT16_MAX) {
                        distances[v] = distances[u] + 1;
                        q.push(v);

                        sum_distances += distances[v];
                        reachable_nodes++;
                    }
                }
            }

            if (sum_distances > 0) {
                closeness_centrality[source] = static_cast<double>(reachable_nodes) / sum_distances;
            }
        }

        return closeness_centrality;
    }

    std::vector<double> compute_betweenness_centrality() const {
        const size_t num_nodes = this->adjacency_list.size();
        std::vector<double> betweenness(num_nodes, 0.0);

        // #pragma omp parallel
        {
            std::vector<double> betweenness_private(num_nodes, 0.0);

            // #pragma omp for schedule(dynamic)
            for (uint16_t s = 0; s < num_nodes; ++s) {
                std::stack<uint16_t> S;
                std::vector<std::vector<uint16_t>> P(num_nodes);
                std::vector<int> sigma(num_nodes, 0);
                std::vector<int> d(num_nodes, -1);
                std::queue<uint16_t> Q;

                sigma[s] = 1;
                d[s] = 0;
                Q.push(s);

                while (!Q.empty()) {
                    uint16_t v = Q.front();
                    Q.pop();
                    S.push(v);

                    for (uint16_t w : this->adjacency_list[v]) {
                        if (d[w] < 0) {
                            d[w] = d[v] + 1;
                            Q.push(w);
                        }
                        if (d[w] == d[v] + 1) {
                            sigma[w] += sigma[v];
                            P[w].push_back(v);
                        }
                    }
                }

                std::vector<double> delta(num_nodes, 0.0);

                while (!S.empty()) {
                    uint16_t w = S.top();
                    S.pop();
                    for (uint16_t v : P[w]) {
                        delta[v] += (static_cast<double>(sigma[v]) / sigma[w]) * (1.0 + delta[w]);
                    }
                    if (w != s) {
                        betweenness_private[w] += delta[w];
                    }
                }
            }

            // #pragma omp critical
            {
                for (uint16_t i = 0; i < num_nodes; ++i) {
                    betweenness[i] += betweenness_private[i];
                }
            }
        }

        // Normalize the betweenness values
        for (double& val : betweenness) {
            val /= 2.0;
        }

        return betweenness;
    }

    std::vector<double> compute_pagerank(const double damping_factor = 0.9, const int max_iterations = 300, const double tol = 1e-8) const {
        const size_t num_nodes = this->adjacency_list.size();
        std::vector pagerank(num_nodes, 1.0 / static_cast<double>(num_nodes));
        std::vector new_pagerank(num_nodes, 0.0);

        std::vector<uint16_t> out_degree(num_nodes);
        for (size_t i = 0; i < num_nodes; ++i) {
            out_degree[i] = static_cast<uint16_t>(this->adjacency_list[i].size());
        }

        for (int iter = 0; iter < max_iterations; ++iter) {
            double dangling_sum = 0.0;
            for (size_t i = 0; i < num_nodes; ++i) {
                new_pagerank[i] = 0.0;
                if (out_degree[i] == 0) {
                    dangling_sum += pagerank[i];
                }
            }

            // #pragma omp parallel for
            for (size_t u = 0; u < num_nodes; ++u) {
                for (uint16_t v : this->adjacency_list[u]) {
                    // #pragma omp atomic
                    new_pagerank[v] += pagerank[u] / out_degree[u];
                }
            }

            double diff = 0.0;
            for (size_t i = 0; i < num_nodes; ++i) {
                new_pagerank[i] = (1.0 - damping_factor) / num_nodes + damping_factor * (new_pagerank[i] + dangling_sum / num_nodes);
                diff += std::abs(new_pagerank[i] - pagerank[i]);
                pagerank[i] = new_pagerank[i];
            }

            if (diff < tol) {
                break;
            }
        }

        return pagerank;
    }
};


int main() {
    try {
        omp_set_num_threads(16);
        const Graph graph = Graph::load_graph_from_file("../data/paths-and-graph/adj_list.txt");
        std::cout << "Loaded graph with " << graph.get_num_nodes() << " nodes\n";

        const auto pairs = graph.load_pairs_from_file("../data/paths-and-graph/unique_games.tsv");
        std::cout << "Loaded " << pairs.size() << " pairs to compute statistics for\n";

        // Add timing here
        const auto start = std::chrono::high_resolution_clock::now();

        auto pair_statistics = graph.compute_statistics_for_pairs(pairs, true, true);

        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

        graph.dump_pair_stats_to_file("../data/paths-and-graph/pair_data.tsv", pair_statistics);

        constexpr uint16_t max_print_value = 30;

        // Compute closeness centrality
        const std::vector<double> closeness_centrality = graph.compute_closeness_centrality();

        // Compute betweenness centrality
        const std::vector<double> betweenness_centrality = graph.compute_betweenness_centrality();

        // Compute PageRank
        const std::vector<double> pagerank = graph.compute_pagerank();

        // Create a vector of NodeData objects and dump to file
        std::vector<NodeData> node_data;
        for (uint16_t i = 0; i < graph.get_num_nodes(); ++i) {
            NodeData data;
            data.name = graph.get_node_name(i);
            data.degree = graph.get_node_degree(i);
            data.closeness_centrality = closeness_centrality[i];
            data.betweenness_centrality = betweenness_centrality[i];
            data.pagerank = pagerank[i];
            node_data.push_back(data);
        }

        dump_node_data_to_file("../data/paths-and-graph/node_data.tsv", node_data);

        // Draw some random samples for the pairs and display the results
        for (uint16_t i = 0; i < max_print_value; ++i) {
            const auto pair = static_cast<uint32_t>(rand() % pairs.size());
            const uint16_t source = pairs[pair].first;
            const uint16_t target = pairs[pair].second;

            std::cout << "Shortest path from " << graph.get_node_name(source) << " to " << graph.get_node_name(target)<< ":\n";
            std::cout << pair_statistics[pair].to_string() << '\n';
            std::cout << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
