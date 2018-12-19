//
// Created by wpj on 2018/12/18.
//
#ifndef TRANSE_MODEL
#define TRANSE_MODEL
#include"model.h"
class TransE: public Model{
    int nh;
public:
    TransE(int ne, int nr, int nh,  double eta, double gamma) : Model(eta, gamma) {
        this->nh = nh;

        E = uniform_matrix(ne, nh, -6.0/sqrt(nh), 6.0/sqrt(nh));
        R = uniform_matrix(nr, nh, -6.0/sqrt(nh), 6.0/sqrt(nh));

        for(int i=0;i<E.size();i++){
            l2_normalize(E[i]);
        }

        for(int i=0;i<R.size();i++){
            l2_normalize(R[i]);
        }

        E_g = const_matrix(ne, nh, init_e);
        R_g = const_matrix(nr, nh, init_e);
    }
    double score(int s, int r, int o) const{

        double dot = 0;
        bool L1 = true;
        for (int i = 0; i < nh; i++) {
            if (L1) {
                dot += fabs(E[s][i] + R[r][i] - E[o][i]);
            } else {
                dot += (E[s][i] + R[r][i] - E[o][i]) * (E[s][i] + R[r][i] - E[o][i]);
            }
        }
        return -dot;
    }
    void score_grad(
            int s,
            int r,
            int o,
            vector<double>& d_s,
            vector<double>& d_r,
            vector<double>& d_o) {
        bool L1 = true;

        for (int i = 0; i < nh; i++) {
            double x = 2*(E[s][i] + R[r][i] - E[o][i]);
            if(L1) {
                x = x > 0 ? 1:-1;
            }
            d_s[i] = x;
            d_r[i] = x;
            d_o[i] = -x;
        }
    }

    void train(int s, int r, int o, int ss, int rr, int oo) {
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

            sgd_update(s, r, o, d_s, d_r, d_o, 1);
            sgd_update(ss, rr, oo, d_ss, d_rr, d_oo, -1);

            l2_normalize(E[s]);
            l2_normalize(E[o]);
            l2_normalize(E[ss]);
            l2_normalize(E[oo]);
        }
    }

};
#endif