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

        E = uniform_matrix(ne, nh, -6/sqrt(nh), 6/sqrt(nh));
        R = uniform_matrix(nr, nh, -6/sqrt(nh), 6/sqrt(nh));

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
        return dot;
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
};
#endif