//
// Created by wpj on 2018/12/29.
//

#include"model.h"
#ifndef MANIFOLDE_MODEL
#define MANIFOLDE_MODEL
class ManifoldE : public Model {
public:
    ManifoldE(int ne, int nr, int nh, double eta, double gamma) : Model(eta, gamma) {
        this->nh = nh;

        E = uniform_matrix(ne, nh, -init_b, init_b);
        R = uniform_matrix(nr, nh, -init_b, init_b);
        A.resize(1);
        A[0] = uniform_matrix(1, nr, 0, 0);

        E_g = const_matrix(ne, nh, init_e);
        R_g = const_matrix(nr, nh, init_e);
        A_g.resize(1);
        A_g[0] = const_matrix(1, nr, init_e);
    }

    double score(int s, int r, int o) const {
        double dot = 0;
        for (int i = 0; i < nh; i++)
            dot += E[s][i] * R[r][i] * E[o][i];
        return -fabs(dot-A[0][0][r] * A[0][0][r]);
    }

    void score_grad(
            int s,
            int r,
            int o,
            vector<double>& d_s,
            vector<double>& d_r,
            vector<double>& d_o,
            double& d_a) {

        double sum = 0;
        for (int i = 0; i < nh; i++)
            sum += E[s][i] * R[r][i] * E[o][i];

        double loss = 2* (sum-A[0][0][r]*A[0][0][r]);
        loss = loss > 0 ? 1 : -1;

        for (int i = 0; i < nh; i++) {
            d_s[i] = loss * R[r][i] * E[o][i];
            d_r[i] = loss * E[s][i] * E[o][i];
            d_o[i] = loss * E[s][i] * R[r][i];
        }
        d_a = -loss * 2 * A[0][0][r];
    }

    void sgd_update(
            int s,
            int r,
            int o,
            const vector<double>& d_s,
            const vector<double>& d_r,
            const vector<double>& d_o,
            const double d_a,
            int flag) {

        Model::sgd_update(s, r, o, d_s, d_r, d_o, flag);
        A[0][0][r] -= flag * eta * d_a;
    }


    void adagrad_update(
            int s,
            int r,
            int o,
            const vector<double>& d_s,
            const vector<double>& d_r,
            const vector<double>& d_o,
            const double d_a,
            int flag) {

        Model::adagrad_update(s, r, o, d_s, d_r, d_o, flag);

        A_g[0][0][r] += d_a*d_a;

        A[0][0][r] -= flag * eta * d_a / sqrt(A_g[0][0][r]);
    }


    void train(int s, int r, int o, int ss, int rr, int oo) {
        vector<double> d_s;
        vector<double> d_r;
        vector<double> d_o;

        double d_a = 0.0;

        vector<double> d_ss;
        vector<double> d_rr;
        vector<double> d_oo;

        double d_aa = 0.0;

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

            score_grad(s, r, o, d_s, d_r, d_o, d_a);
            score_grad(ss, rr, oo, d_ss, d_rr, d_oo, d_aa);

            adagrad_update(s, r, o, d_s, d_r, d_o, d_a, 1);
            adagrad_update(ss, rr, oo, d_ss, d_rr, d_oo, d_aa, -1);

            l2_normalize(E[s]);
            l2_normalize(E[o]);

            if(s!=ss)l2_normalize(E[ss]);
            if(o!=oo)l2_normalize(E[oo]);
        }

    }

};
#endif