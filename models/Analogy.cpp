//
// Created by wpj on 2018/12/18.
//

#ifndef ANALOGY_MODEL
#define ANALOGY_MODEL
#include"model.h"

class Analogy : public Model {
    int nh1;
    int nh2;

public:
    Analogy(int ne, int nr, int nh, int num_scalar, double eta, double gamma) : Model(eta, gamma) {
        this->nh1 = num_scalar;
        this->nh2 = nh - num_scalar;
        assert( this->nh2 % 2 == 0 );

        E = uniform_matrix(ne, nh, -init_b, init_b);
        R = uniform_matrix(nr, nh, -init_b, init_b);
        E_g = const_matrix(ne, nh, init_e);
        R_g = const_matrix(nr, nh, init_e);
    }

    double score(int s, int r, int o) const {
        double dot = 0;

        int i = 0;
        for (; i < nh1; i++)
            dot += E[s][i] * R[r][i] * E[o][i];

        int nh2_2 = nh2/2;
        for (; i < nh1 + nh2_2; i++) {
            dot += R[r][i]       * E[s][i]       * E[o][i];
            dot += R[r][i]       * E[s][nh2_2+i] * E[o][nh2_2+i];
            dot += R[r][nh2_2+i] * E[s][i]       * E[o][nh2_2+i];
            dot -= R[r][nh2_2+i] * E[s][nh2_2+i] * E[o][i];
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

        int i = 0;
        for (; i < nh1; i++) {
            d_s[i] = R[r][i] * E[o][i];
            d_r[i] = E[s][i] * E[o][i];
            d_o[i] = E[s][i] * R[r][i];
        }

        int nh2_2 = nh2/2;
        for (; i < nh1 + nh2_2; i++) {
            // re
            d_s[i] = R[r][i] * E[o][i] + R[r][nh2_2+i] * E[o][nh2_2+i];
            d_r[i] = E[s][i] * E[o][i] + E[s][nh2_2+i] * E[o][nh2_2+i];
            d_o[i] = R[r][i] * E[s][i] - R[r][nh2_2+i] * E[s][nh2_2+i];
            // im
            d_s[nh2_2+i] = R[r][i] * E[o][nh2_2+i] - R[r][nh2_2+i] * E[o][i];
            d_r[nh2_2+i] = E[s][i] * E[o][nh2_2+i] - E[s][nh2_2+i] * E[o][i];
            d_o[nh2_2+i] = R[r][i] * E[s][nh2_2+i] + R[r][nh2_2+i] * E[s][i];
        }

    }
};
#endif