#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif

#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;


template <typename T>
class ConcurrentQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cvNotEmpty_;
    bool stopped_ = false;

public:
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cvNotEmpty_.notify_one();
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cvNotEmpty_.wait(lock, [this]() { return !queue_.empty() || stopped_; });
        if (queue_.empty() && stopped_) {
            return false;
        }
        out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cvNotEmpty_.notify_all();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = false;
        while (!queue_.empty()) queue_.pop();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

class ThreadPool {
private:
    std::vector<std::thread> workers_;
    ConcurrentQueue<std::function<void()>> taskQueue_;
    std::atomic<bool> stopFlag_{false};
    std::atomic<size_t> activeTasks_{0};
    std::mutex waitMutex_;
    std::condition_variable cvWait_;

public:
    ThreadPool(size_t threads = std::thread::hardware_concurrency()) {
        resize(threads);
    }

    ~ThreadPool() {
        shutdown();
    }

    void submit(std::function<void()> task) {
        activeTasks_++;
        taskQueue_.push([this, task = std::move(task)]() {
            try {
                task();
            } catch (...) {}
            size_t remaining = --activeTasks_;
            if (remaining == 0) {
                std::lock_guard<std::mutex> lock(waitMutex_);
                cvWait_.notify_all();
            }
        });
    }

    void waitAll() {
        std::unique_lock<std::mutex> lock(waitMutex_);
        cvWait_.wait(lock, [this]() { return activeTasks_ == 0; });
    }

    void resize(size_t threads) {
        shutdown();
        taskQueue_.reset();
        stopFlag_ = false;
        if (threads == 0) threads = 1;
        for (size_t i = 0; i < threads; ++i) {
            workers_.emplace_back([this]() {
                std::function<void()> task;
                while (taskQueue_.pop(task)) {
                    task();
                }
            });
        }
    }

    void shutdown() {
        stopFlag_ = true;
        taskQueue_.stop();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }
};


struct Posting {
    uint32_t docId;
    uint32_t termFreq;
    std::vector<uint32_t> positions;
};

struct DocumentRecord {
    uint32_t id;
    std::string filename;
    std::string path;
    size_t totalWords = 0;
    std::string snippetPreview;
};

class IndexShard {
private:
    mutable std::shared_mutex rwMutex_;
    std::map<std::string, std::vector<Posting>> indexMap_;

public:
    void addPosting(const std::string& term, uint32_t docId, uint32_t position) {
        std::unique_lock<std::shared_mutex> lock(rwMutex_);
        auto& postings = indexMap_[term];
        if (!postings.empty() && postings.back().docId == docId) {
            postings.back().termFreq++;
            postings.back().positions.push_back(position);
        } else {
            postings.push_back({docId, 1, {position}});
        }
    }

    std::vector<Posting> getPostings(const std::string& term) const {
        std::shared_lock<std::shared_mutex> lock(rwMutex_);
        auto it = indexMap_.find(term);
        if (it != indexMap_.end()) {
            return it->second;
        }
        return {};
    }

    size_t getTermCount() const {
        std::shared_lock<std::shared_mutex> lock(rwMutex_);
        return indexMap_.size();
    }

    void clear() {
        std::unique_lock<std::shared_mutex> lock(rwMutex_);
        indexMap_.clear();
    }
};

class ThreadSafeInvertedIndex {
public:
    static constexpr size_t NUM_SHARDS = 128;

private:
    std::vector<IndexShard> shards_{NUM_SHARDS};
    mutable std::shared_mutex docRegistryMutex_;
    std::vector<DocumentRecord> documents_;
    std::atomic<size_t> totalIndexedTokens_{0};

    size_t getShardIndex(const std::string& term) const {
        std::hash<std::string> hasher;
        return hasher(term) % NUM_SHARDS;
    }

public:
    void clear() {
        for (auto& shard : shards_) {
            shard.clear();
        }
        std::unique_lock<std::shared_mutex> lock(docRegistryMutex_);
        documents_.clear();
        totalIndexedTokens_ = 0;
    }

    uint32_t registerDocument(const std::string& filename, const std::string& path, size_t totalWords, const std::string& snippet) {
        std::unique_lock<std::shared_mutex> lock(docRegistryMutex_);
        uint32_t docId = static_cast<uint32_t>(documents_.size());
        documents_.push_back({docId, filename, path, totalWords, snippet});
        totalIndexedTokens_ += totalWords;
        return docId;
    }

    void insertPosting(const std::string& term, uint32_t docId, uint32_t position) {
        size_t shard = getShardIndex(term);
        shards_[shard].addPosting(term, docId, position);
    }

    std::vector<Posting> lookupTerm(const std::string& term) const {
        size_t shard = getShardIndex(term);
        return shards_[shard].getPostings(term);
    }

    DocumentRecord getDocument(uint32_t docId) const {
        std::shared_lock<std::shared_mutex> lock(docRegistryMutex_);
        if (docId < documents_.size()) {
            return documents_[docId];
        }
        return {};
    }

    size_t getDocumentCount() const {
        std::shared_lock<std::shared_mutex> lock(docRegistryMutex_);
        return documents_.size();
    }

    size_t getTotalTokens() const {
        return totalIndexedTokens_.load();
    }

    size_t getTotalUniqueTerms() const {
        size_t count = 0;
        for (const auto& shard : shards_) {
            count += shard.getTermCount();
        }
        return count;
    }
};


class TextTokenizer {
public:
    static std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::string current;
        for (char c : text) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                current += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            } else if (!current.empty()) {
                if (current.length() >= 2) {
                    tokens.push_back(current);
                }
                current.clear();
            }
        }
        if (current.length() >= 2) {
            tokens.push_back(current);
        }
        return tokens;
    }
};


struct SearchResultItem {
    std::string docId;
    double score;
    std::string snippet;
};

class BM25Scorer {
public:
    static std::vector<SearchResultItem> rank(
        const ThreadSafeInvertedIndex& index,
        const std::vector<std::string>& queryTerms,
        size_t topK = 10
    ) {
        size_t totalDocs = index.getDocumentCount();
        if (totalDocs == 0 || queryTerms.empty()) return {};

        double avgDocLength = static_cast<double>(index.getTotalTokens()) / totalDocs;
        if (avgDocLength <= 0.0) avgDocLength = 1.0;

        constexpr double k1 = 1.2;
        constexpr double b = 0.75;

        std::unordered_map<uint32_t, double> docScores;

        for (const auto& term : queryTerms) {
            auto postings = index.lookupTerm(term);
            size_t docFreq = postings.size();
            if (docFreq == 0) continue;

            double idf = std::log(1.0 + (totalDocs - docFreq + 0.5) / (docFreq + 0.5));
            if (idf < 0) idf = 0.01;

            for (const auto& posting : postings) {
                auto doc = index.getDocument(posting.docId);
                double docLen = static_cast<double>(doc.totalWords);
                double tf = static_cast<double>(posting.termFreq);

                double termScore = idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (docLen / avgDocLength)));
                docScores[posting.docId] += termScore;
            }
        }

        std::vector<std::pair<uint32_t, double>> scoreList(docScores.begin(), docScores.end());
        std::sort(scoreList.begin(), scoreList.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        std::vector<SearchResultItem> results;
        for (size_t i = 0; i < std::min(topK, scoreList.size()); ++i) {
            auto doc = index.getDocument(scoreList[i].first);
            results.push_back({doc.filename, scoreList[i].second, doc.snippetPreview});
        }
        return results;
    }
};


class OllamaClient {
private:
    std::string hostUrl_;

public:
    OllamaClient(std::string hostUrl = "http://ollama:11434") : hostUrl_(std::move(hostUrl)) {}

    bool checkHealth() {
        try {
            std::string host = hostUrl_;
            if (host.rfind("http://", 0) == 0) {
                host = host.substr(7);
            }
            size_t colonPos = host.find(':');
            std::string ip = (colonPos != std::string::npos) ? host.substr(0, colonPos) : host;
            int port = (colonPos != std::string::npos) ? std::stoi(host.substr(colonPos + 1)) : 11434;

            httplib::Client cli(ip, port);
            cli.set_connection_timeout(1, 0); // 1 sec
            auto res = cli.Get("/api/tags");
            return (res && res->status == 200);
        } catch (...) {
            return false;
        }
    }
};


int main(int argc, char** argv) {
    const char* envPort = std::getenv("PORT");
    int port = envPort ? std::stoi(envPort) : 8080;

    const char* envOllama = std::getenv("OLLAMA_HOST");
    std::string ollamaHost = envOllama ? envOllama : "http://ollama:11434";

    const char* envDataset = std::getenv("DATASET_PATH");
    std::string datasetPath = envDataset ? envDataset : "./data/dataset";

    ThreadSafeInvertedIndex invertedIndex;
    ThreadPool threadPool;
    OllamaClient ollama(ollamaHost);

    std::atomic<size_t> lastBuildTimeMs{0};
    std::atomic<size_t> lastThreadsUsed{1};

    httplib::Server svr;

    // 1. GET /api/health (Mandatory criterion for 1st Attestation)
    svr.Get("/api/health", [&](const httplib::Request&, httplib::Response& res) {
        bool ollamaAlive = ollama.checkHealth();
        json response = {
            {"status", "ok"},
            {"ollama", ollamaAlive}
        };
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // 2. GET /api/stats
    svr.Get("/api/stats", [&](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"documents", invertedIndex.getDocumentCount()},
            {"index_build_time_ms", lastBuildTimeMs.load()},
            {"threads_used", lastThreadsUsed.load()},
            {"total_tokens", invertedIndex.getTotalTokens()},
            {"unique_terms", invertedIndex.getTotalUniqueTerms()}
        };
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // 3. POST /api/index
    svr.Post("/api/index", [&](const httplib::Request& req, httplib::Response& res) {
        size_t threads = 4;
        try {
            if (!req.body.empty()) {
                auto parsed = json::parse(req.body);
                if (parsed.contains("threads")) {
                    threads = parsed["threads"].get<size_t>();
                }
            }
        } catch (...) {}

        if (threads == 0) threads = 1;
        lastThreadsUsed = threads;

        auto startTime = std::chrono::high_resolution_clock::now();
        invertedIndex.clear();
        threadPool.resize(threads);

        // Scan files in dataset directory
        std::vector<std::string> filePaths;
        if (fs::exists(datasetPath)) {
            for (const auto& entry : fs::recursive_directory_iterator(datasetPath)) {
                if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                    filePaths.push_back(entry.path().string());
                }
            }
        }

        // Parallel indexing via ThreadPool
        for (const auto& filePath : filePaths) {
            threadPool.submit([&invertedIndex, filePath]() {
                std::ifstream f(filePath);
                if (!f.is_open()) return;

                std::stringstream buffer;
                buffer << f.rdbuf();
                std::string content = buffer.str();

                std::string snippet = content.substr(0, std::min(content.length(), size_t(160)));
                std::string filename = fs::path(filePath).filename().string();

                auto tokens = TextTokenizer::tokenize(content);
                uint32_t docId = invertedIndex.registerDocument(filename, filePath, tokens.size(), snippet);

                for (uint32_t pos = 0; pos < tokens.size(); ++pos) {
                    invertedIndex.insertPosting(tokens[pos], docId, pos);
                }
            });
        }

        threadPool.waitAll();

        auto endTime = std::chrono::high_resolution_clock::now();
        size_t durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        lastBuildTimeMs = durationMs;

        json response = {
            {"status", "ok"},
            {"documents_indexed", invertedIndex.getDocumentCount()},
            {"time_ms", durationMs}
        };
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // 4. POST /api/search
    svr.Post("/api/search", [&](const httplib::Request& req, httplib::Response& res) {
        std::string queryText;
        std::string mode = "keyword";

        try {
            auto parsed = json::parse(req.body);
            if (parsed.contains("query")) queryText = parsed["query"].get<std::string>();
            if (parsed.contains("mode")) mode = parsed["mode"].get<std::string>();
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error": "Invalid JSON"})", "application/json");
            return;
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        auto queryTerms = TextTokenizer::tokenize(queryText);
        auto searchResults = BM25Scorer::rank(invertedIndex, queryTerms, 10);
        auto endTime = std::chrono::high_resolution_clock::now();
        size_t durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

        json resultsArr = json::array();
        for (const auto& r : searchResults) {
            resultsArr.push_back({
                {"doc_id", r.docId},
                {"score", std::round(r.score * 1000.0) / 1000.0},
                {"snippet", r.snippet}
            });
        }

        json response = {
            {"results", resultsArr},
            {"time_ms", durationMs}
        };
        res.set_content(response.dump(), "application/json");
        res.status = 200;
    });

    // Static Web UI mount
    if (fs::exists("./static")) {
        svr.set_mount_point("/", "./static");
    }

    std::cout << "========================================================\n";
    std::cout << " Listening on http://0.0.0.0:" << port << "\n";
    std::cout << " Ollama host: " << ollamaHost << "\n";
    std::cout << " Dataset dir: " << datasetPath << "\n";
    std::cout << "========================================================\n";

    svr.listen("0.0.0.0", port);
    return 0;
}
