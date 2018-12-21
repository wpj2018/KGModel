//
// Created by wpj on 2018/12/18.
//
#ifndef BASE_MODEL
#define BASE_MODEL

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <random>
#include <omp.h>
#include <cassert>
#include <cstring>
#include <iterator>

using namespace std;

static default_random_engine GLOBAL_GENERATOR;
static uniform_real_distribution<double> UNIFORM(0, 1);

typedef tuple<int, int, int> triplet;

vector<string> read_first_column(const string& fname) {
    ifstream ifs(fname, ios::in);

    string line;
    string item;
    vector<string> items;

    assert(!ifs.fail());

    while (getline(ifs, line)) {
        stringstream ss(line);
        ss >> item;
        items.push_back(item);
    }
    ifs.close();

    return items;
}

unordered_map<string, int> create_id_mapping(const vector<string>& items) {
    unordered_map<string, int> map;

    for (int i = 0; i < (int) items.size(); i++)
        map[items[i]] = i;

    return map;
}

vector<triplet> create_sros(
        const string& fname,
        const unordered_map<string, int>& ent_map,
        const unordered_map<string, int>& rel_map) {

    ifstream ifs(fname, ios::in);

    string line;
    string s, r, o;
    vector<triplet> sros;

    assert(!ifs.fail());

    while (getline(ifs, line)) {
        stringstream ss(line);
        ss >> s >> r >> o;
        sros.push_back( make_tuple(ent_map.at(s), rel_map.at(r), ent_map.at(o)) );
    }
    ifs.close();

    return sros;
}

vector<vector<double>> uniform_matrix(int m, int n, double l, double h) {
    vector<vector<double>> matrix;
    matrix.resize(m);
    for (int i = 0; i < m; i++)
        matrix[i].resize(n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = (h-l)*UNIFORM(GLOBAL_GENERATOR) + l;

    return matrix;
}


vector<vector<double>> const_matrix(int m, int n, double c) {
    vector<vector<double>> matrix;
    matrix.resize(m);
    for (int i = 0; i < m; i++)
        matrix[i].resize(n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = c;

    return matrix;
}

vector<double>matmul(const vector<vector<double> > &m, const vector<double>& a){
    int rows = m.size();
    int cols = m[0].size();

    vector<double>res(rows, 0);
    for(unsigned int i=0; i < rows; i++){
        double tmp = 0;
        int j = 0;
        for(; j+4 < cols; j+=4){
             tmp += m[i][j] * a[j] + m[i][j+1] * a[j+1] + m[i][j+2] * a[j+2] + m[i][j+3] * a[j+3];
        }
        for(; j < cols; j++){
            tmp += m[i][j] * a[j];
        }

        res[i] = tmp;
    }
    return res;
}

vector<vector<double>>T(const vector<vector<double>> &m){
    int rows = m.size();
    assert(rows>=1);
    int cols = m[0].size();
    vector<vector<double>>res = const_matrix(cols, rows, 0);
    for(unsigned int i=0;i<res.size();i++){
        for(unsigned int j=0;j<res[i].size();j++){
            res[i][j] = m[j][i];
        }
    }
    return res;
}
vector<double>matmul(const vector<double>& a, const vector<vector<double> > &m){
    vector<double>res = matmul(T(m), a);
    return res;
}

double matmul(const vector<double>& a, const vector<double> &b){
    assert(a.size()==b.size());
    double res = 0;
    for(unsigned int i=0;i<a.size();i++){
        res+=a[i]*b[i];
    }
    return res;
}
double length(const vector<double>&a, int l){
    double sum = 0;
    for(unsigned int i=0;i<a.size();i++){
        if(l==1)sum+=fabs(a[i]);
        else if(l==2)sum+=a[i]*a[i];
        else{
            throw ("not implement");
        }
    }
    return sum;
}
vector<double>abs(const vector<double>&a){
    vector<double>res(a);
    for(unsigned int i=0;i<a.size();i++){
        res[i] = fabs(a[i]);
    }
    return res;
}

vector<double>sub(const vector<double>&a, const vector<double>&b){
    vector<double>res(a.size());
    for(unsigned int i=0;i<res.size();i++){
        res[i] = a[i]-b[i];
    }
    return res;
}

vector<double>sub(const vector<double>&a, const vector<double>&b, const vector<double>&c){
    vector<double>res = sub(sub(a, b),c);
    return res;
}

vector<int> range(int n) {  // 0 ... n-1
    vector<int> v;
    v.reserve(n);
    for (int i = 0; i < n; i++)
        v.push_back(i);
    return v;
}

void l2_normalize(vector<double>& vec) {
    double sq_norm = 0;
    for (unsigned int i = 0; i < vec.size(); i++)
        sq_norm += vec[i] * vec[i];
    double norm = sqrt(sq_norm);
    for (unsigned int i = 0; i < vec.size(); i++)
        vec[i] /= norm;
}

double sigmoid(double x, double cutoff=30) {
    if (x > +cutoff) return 1.;
    if (x < -cutoff) return 0.;
    return 1./(1.+exp(-x));
}

class SROBucket {
    unordered_set<int64_t> __sros;
    unordered_map<int64_t, vector<int>> __sr2o;
    unordered_map<int64_t, vector<int>> __or2s;

    int64_t hash(int a, int b, int c) const {
        int64_t x = a;
        x = (x << 20) + b;
        return (x << 20) + c;
    }

    int64_t hash(int a, int b) const {
        int64_t x = a;
        return (x << 32) + b;
    }

public:
    SROBucket(const vector<triplet>& sros) {
        for (auto sro : sros) {
            int s = get<0>(sro);
            int r = get<1>(sro);
            int o = get<2>(sro);

            int64_t __sro = hash(s, r, o);
            __sros.insert(__sro);

            int64_t __sr = hash(s, r);
            if (__sr2o.find(__sr) == __sr2o.end())
                __sr2o[__sr] = vector<int>();
            __sr2o[__sr].push_back(o);

            int64_t __or = hash(o, r);
            if (__or2s.find(__or) == __or2s.end())
                __or2s[__or] = vector<int>();
            __or2s[__or].push_back(s);
        }
    }

    bool contains(int a, int b, int c) const {
        return __sros.find( hash(a, b, c) ) != __sros.end();
    }

    vector<int> sr2o(int s, int r) const {
        return __sr2o.at(hash(s,r));
    }

    vector<int> or2s(int o, int r) const {
        return __or2s.at(hash(o,r));
    }
};

// try sample pairs
class NegativeSampler {
    uniform_int_distribution<int> unif_e;
    uniform_int_distribution<int> unif_r;
    default_random_engine generator;

public:
    NegativeSampler(int ne, int nr, int seed) :
            unif_e(0, ne-1), unif_r(0, nr-1), generator(seed) {}

    int random_entity() {
        return unif_e(generator);
    }

    int random_relation() {
        return unif_r(generator);
    }
};

class Model {

protected:
    double eta;
    double gamma;
    const double init_b = 1e-2;
    const double init_e = 1e-6;

    vector<vector<double>> E;
    vector<vector<double>> R;
    vector<vector<double>> E_g;
    vector<vector<double>> R_g;

public:

    Model(double eta, double gamma) {
        this->eta = eta;
        this->gamma = gamma;
    }

    void save(const string& fname) {
        ofstream ofs(fname, ios::out);

        for (unsigned int i = 0; i < E.size(); i++) {
            for (unsigned int j = 0; j < E[i].size(); j++)
                ofs << E[i][j] << ' ';
            ofs << endl;
        }

        for (unsigned int i = 0; i < R.size(); i++) {
            for (unsigned int j = 0; j < R[i].size(); j++)
                ofs << R[i][j] << ' ';
            ofs << endl;
        }

        ofs.close();
    }

    void load(const string& fname) {
        ifstream ifs(fname, ios::in);

        for (unsigned int i = 0; i < E.size(); i++)
            for (unsigned int j = 0; j < E[i].size(); j++)
                ifs >> E[i][j];

        for (unsigned int i = 0; i < R.size(); i++)
            for (unsigned int j = 0; j < R[i].size(); j++)
                ifs >> R[i][j];

        ifs.close();
    }

    void adagrad_update(
            int s,
            int r,
            int o,
            const vector<double>& d_s,
            const vector<double>& d_r,
            const vector<double>& d_o,
            int flag) {

        for (unsigned int i = 0; i < E[s].size(); i++) E_g[s][i] += d_s[i] * d_s[i];
        for (unsigned int i = 0; i < R[r].size(); i++) R_g[r][i] += d_r[i] * d_r[i];
        for (unsigned int i = 0; i < E[o].size(); i++) E_g[o][i] += d_o[i] * d_o[i];

        for (unsigned int i = 0; i < E[s].size(); i++) E[s][i] -= flag * eta * d_s[i] / sqrt(E_g[s][i]);
        for (unsigned int i = 0; i < R[r].size(); i++) R[r][i] -= flag * eta * d_r[i] / sqrt(R_g[r][i]);
        for (unsigned int i = 0; i < E[o].size(); i++) E[o][i] -= flag * eta * d_o[i] / sqrt(E_g[o][i]);
    }

    void sgd_update(
            int s,
            int r,
            int o,
            const vector<double>& d_s,
            const vector<double>& d_r,
            const vector<double>& d_o,
            int flag) {

        for (unsigned int i = 0; i < E[s].size(); i++) E[s][i] -= flag * eta * d_s[i];
        for (unsigned int i = 0; i < R[r].size(); i++) R[r][i] -= flag * eta * d_r[i];
        for (unsigned int i = 0; i < E[o].size(); i++) E[o][i] -= flag * eta * d_o[i];
    }

    void train(int s, int r, int o, bool is_positive) {
        vector<double> d_s;
        vector<double> d_r;
        vector<double> d_o;

        d_s.resize(E[s].size());
        d_r.resize(R[r].size());
        d_o.resize(E[o].size());

        double offset = is_positive ? 1 : 0;
        double d_loss = sigmoid(score(s, r, o)) - offset;

        score_grad(s, r, o, d_s, d_r, d_o);

        for (unsigned int i = 0; i < d_s.size(); i++) d_s[i] *= d_loss;
        for (unsigned int i = 0; i < d_r.size(); i++) d_r[i] *= d_loss;
        for (unsigned int i = 0; i < d_o.size(); i++) d_o[i] *= d_loss;

        double gamma_s = gamma / d_s.size();
        double gamma_r = gamma / d_r.size();
        double gamma_o = gamma / d_o.size();

        for (unsigned int i = 0; i < d_s.size(); i++) d_s[i] += gamma_s * E[s][i];
        for (unsigned int i = 0; i < d_r.size(); i++) d_r[i] += gamma_r * R[r][i];
        for (unsigned int i = 0; i < d_o.size(); i++) d_o[i] += gamma_o * E[o][i];

        adagrad_update(s, r, o, d_s, d_r, d_o, 1);
    }

    virtual void train(int s, int r, int o, int ss, int rr, int oo) {
        vector<double> d_s;
        vector<double> d_r;
        vector<double> d_o;

        vector<double> d_ss;
        vector<double> d_rr;
        vector<double> d_oo;

        d_s.resize(E[s].size());
        d_r.resize(R[r].size());
        d_o.resize(E[o].size());

        d_ss.resize(E[ss].size());
        d_rr.resize(R[rr].size());
        d_oo.resize(E[oo].size());


        double sum1 = -score(s, r, o);
        double sum2 = -score(ss, rr, oo);
        double margin = 1.0;
        if (sum1+margin > sum2)
        {
            score_grad(s, r, o, d_s, d_r, d_o);
            score_grad(ss, rr, oo, d_ss, d_rr, d_oo);

            adagrad_update(s, r, o, d_s, d_r, d_o, 1);
            adagrad_update(ss, rr, oo, d_ss, d_rr, d_oo, -1);

            l2_normalize(E[s]);
            l2_normalize(E[o]);
            l2_normalize(E[ss]);
            l2_normalize(E[oo]);
        }
    }

    virtual double score(int s, int r, int o) const = 0;

    virtual void score_grad(
            int s,
            int r,
            int o,
            vector<double>& d_s,
            vector<double>& d_r,
            vector<double>& d_o) {};
};

class Evaluator {
    int ne;
    int nr;
    const vector<triplet>& sros;
    const SROBucket& sro_bucket;

public:
    Evaluator(int ne, int nr, const vector<triplet>& sros, const SROBucket& sro_bucket) :
            ne(ne), nr(nr), sros(sros), sro_bucket(sro_bucket) {}

    unordered_map<string, double> evaluate(const Model *model, int truncate) {
        int N = this->sros.size();

        if (truncate > 0)
            N = min(N, truncate);

        double mrr_s = 0.;
        double mrr_r = 0.;
        double mrr_o = 0.;

        double mrr_s_raw = 0.;
        double mrr_o_raw = 0.;

        double mr_s = 0.;
        double mr_r = 0.;
        double mr_o = 0.;

        double mr_s_raw = 0.;
        double mr_o_raw = 0.;

        double hits01_s = 0.;
        double hits01_r = 0.;
        double hits01_o = 0.;

        double hits03_s = 0.;
        double hits03_r = 0.;
        double hits03_o = 0.;

        double hits10_s = 0.;
        double hits10_r = 0.;
        double hits10_o = 0.;

#pragma omp parallel for reduction(+: mrr_s, mrr_r, mrr_o, mr_s, mr_r, mr_o, \
                hits01_s, hits01_r, hits01_o, hits03_s, hits03_r, hits03_o, hits10_s, hits10_r, hits10_o)
        for (int i = 0; i < N; i++) {
            auto ranks = this->rank(model, sros[i]);

            double rank_s = get<0>(ranks);
            double rank_r = get<1>(ranks);
            double rank_o = get<2>(ranks);
            double rank_s_raw = get<3>(ranks);
            double rank_o_raw = get<4>(ranks);

            mrr_s += 1./rank_s;
            mrr_r += 1./rank_r;
            mrr_o += 1./rank_o;
            mrr_s_raw += 1./rank_s_raw;
            mrr_o_raw += 1./rank_o_raw;

            mr_s += rank_s;
            mr_r += rank_r;
            mr_o += rank_o;
            mr_s_raw += rank_s_raw;
            mr_o_raw += rank_o_raw;

            hits01_s += rank_s <= 01;
            hits01_r += rank_r <= 01;
            hits01_o += rank_o <= 01;

            hits03_s += rank_s <= 03;
            hits03_r += rank_r <= 03;
            hits03_o += rank_o <= 03;

            hits10_s += rank_s <= 10;
            hits10_r += rank_r <= 10;
            hits10_o += rank_o <= 10;
        }

        unordered_map<string, double> info;

        info["mrr_s"] = mrr_s / N;
        info["mrr_r"] = mrr_r / N;
        info["mrr_o"] = mrr_o / N;
        info["mrr_s_raw"] = mrr_s_raw / N;
        info["mrr_o_raw"] = mrr_o_raw / N;

        info["mr_s"] = mr_s / N;
        info["mr_r"] = mr_r / N;
        info["mr_o"] = mr_o / N;
        info["mr_s_raw"] = mr_s_raw / N;
        info["mr_o_raw"] = mr_o_raw / N;

        info["hits01_s"] = hits01_s / N;
        info["hits01_r"] = hits01_r / N;
        info["hits01_o"] = hits01_o / N;

        info["hits03_s"] = hits03_s / N;
        info["hits03_r"] = hits03_r / N;
        info["hits03_o"] = hits03_o / N;

        info["hits10_s"] = hits10_s / N;
        info["hits10_r"] = hits10_r / N;
        info["hits10_o"] = hits10_o / N;

        return info;
    }

private:

    tuple<double, double, double, double, double> rank(const Model *model, const triplet& sro) {
        int rank_s = 1;
        int rank_r = 1;
        int rank_o = 1;

        int s = get<0>(sro);
        int r = get<1>(sro);
        int o = get<2>(sro);

        // XXX:
        // There might be degenerated cases when all output scores == 0, leading to perfect but meaningless results.
        // A quick fix is to add a small offset to the base_score.
        double base_score = model->score(s, r, o) - 1e-32;

        for (int ss = 0; ss < ne; ss++)
            if (model->score(ss, r, o) > base_score) rank_s++;

        for (int rr = 0; rr < nr; rr++)
            if (model->score(s, rr, o) > base_score) rank_r++;

        for (int oo = 0; oo < ne; oo++)
            if (model->score(s, r, oo) > base_score) rank_o++;

        int rank_s_raw = rank_s;
        int rank_o_raw = rank_o;

        for (auto ss : sro_bucket.or2s(o, r))
            if (model->score(ss, r, o) > base_score) rank_s--;

        for (auto oo : sro_bucket.sr2o(s, r))
            if (model->score(s, r, oo) > base_score) rank_o--;

        return make_tuple(rank_s, rank_r, rank_o, rank_s_raw, rank_o_raw);
    }
};

void pretty_print(const char* prefix, const unordered_map<string, double>& info) {
    printf("%s  MRR    \t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("mrr_s"),    100*info.at("mrr_r"),    100*info.at("mrr_o"));
    printf("%s  MRR_RAW\t%.2f\t%.2f\n", prefix, 100*info.at("mrr_s_raw"),    100*info.at("mrr_o_raw"));
    printf("%s  MR     \t%.2f\t%.2f\t%.2f\n", prefix, info.at("mr_s"), info.at("mr_r"), info.at("mr_o"));
    printf("%s  MR_RAW \t%.2f\t%.2f\n", prefix, info.at("mr_s_raw"), info.at("mr_o_raw"));
    printf("%s  Hits@01\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits01_s"), 100*info.at("hits01_r"), 100*info.at("hits01_o"));
    printf("%s  Hits@03\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits03_s"), 100*info.at("hits03_r"), 100*info.at("hits03_o"));
    printf("%s  Hits@10\t%.2f\t%.2f\t%.2f\n", prefix, 100*info.at("hits10_s"), 100*info.at("hits10_r"), 100*info.at("hits10_o"));
}


#endif
