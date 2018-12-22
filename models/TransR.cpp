//
// Created by wpj on 2018/12/21.
//

#ifndef TRANSR_MODEL
#define TRANSR_MODEL
#include"model.h"
#include<time.h>
class TransR: public Model{
    int nh;
public:
    TransR(int ne, int nr, int nh,  double eta, double gamma) : Model(eta, gamma) {
        this->nh = nh;

        E = uniform_matrix(ne, nh, -6.0/sqrt(nh), 6.0/sqrt(nh));
        R = uniform_matrix(nr, nh, -6.0/sqrt(nh), 6.0/sqrt(nh));

        for(unsigned int i=0;i<E.size();i++){
            l2_normalize(E[i]);
        }

        for(unsigned int i=0;i<R.size();i++){
            l2_normalize(R[i]);
        }

        A.resize(nr);
        for(unsigned int i=0;i<A.size();i++){
            vector<vector<double>> tmp = const_matrix(nh, nh, 0);
            for(unsigned int j=0; j < tmp.size(); j++){
                tmp[j][j] = 1;
            }
            A[i] = tmp;
        }

        E_g = const_matrix(ne, nh, init_e);
        R_g = const_matrix(nr, nh, init_e);
        A_g = const_matrix(nh, nh, init_e);
    }

    double score(int s, int r, int o) const{
        vector<double>s_ = matmul(A[r], E[s]);
        vector<double>o_ = matmul(A[r], E[o]);
        double sum = length(sub(o_, s_, R[r]), 2);
        return -sum;
    }

    void score_grad(
            int s,
            int r,
            int o,
            vector<double>& d_s,
            vector<double>& d_r,
            vector<double>& d_o,
            double* d_a) {


        vector<double>s_ = matmul(A[r], E[s]);
        vector<double>o_ = matmul(A[r], E[o]);
        vector<double>diff = sub(o_, s_, R[r]);


        for(int j =0; j < nh;j++){
            double sum = 0;
            for(int i=0; i < nh; i++){
                sum += 2 * diff[i] * A[r][i][j];
            }
            d_s[j] = -sum;
            d_r[j] = -diff[j];
            d_o[j] = sum;
        }

        for(int i=0; i<nh; i++){
            for(int j = 0; j < nh; j++){
                d_a[i*nh+j] = 2 * diff[i] * (E[o][j] - E[s][j]);
            }
        }

    }


    void sgd_update(
            int s,
            int r,
            int o,
            const vector<double>& d_s,
            const vector<double>& d_r,
            const vector<double>& d_o,
            double* d_a,
            int flag) {

        for (unsigned int i = 0; i < E[s].size(); i++) E[s][i] -= flag * eta * d_s[i];
        for (unsigned int i = 0; i < R[r].size(); i++) R[r][i] -= flag * eta * d_r[i];
        for (unsigned int i = 0; i < E[o].size(); i++) E[o][i] -= flag * eta * d_o[i];

        for (int i = 0; i < nh; i++){
            for(int j = 0; j < nh; j++){
                A[r][i][j] -= flag * eta * d_a[i* nh +j];
            }
        }
    }

    void adagrad_update(
            int s,
            int r,
            int o,
            const vector<double>& d_s,
            const vector<double>& d_r,
            const vector<double>& d_o,
            const double* d_a,
            int flag) {

        for (unsigned int i = 0; i < E[s].size(); i++) E_g[s][i] += d_s[i] * d_s[i];
        for (unsigned int i = 0; i < R[r].size(); i++) R_g[r][i] += d_r[i] * d_r[i];
        for (unsigned int i = 0; i < E[o].size(); i++) E_g[o][i] += d_o[i] * d_o[i];

        for (int i = 0; i < nh; i++){
            for(int j = 0; j < nh; j++){
                A_g[i][j] += d_a[i* nh +j] * d_a[i * nh + j];
            }
        }

        for (unsigned int i = 0; i < E[s].size(); i++) E[s][i] -= flag * eta * d_s[i] / sqrt(E_g[s][i]);
        for (unsigned int i = 0; i < R[r].size(); i++) R[r][i] -= flag * eta * d_r[i] / sqrt(R_g[r][i]);
        for (unsigned int i = 0; i < E[o].size(); i++) E[o][i] -= flag * eta * d_o[i] / sqrt(E_g[o][i]);

        for (int  i = 0; i < nh; i++){
            for(int j = 0; j < nh; j++){
                A[r][i][j] -= flag * eta * d_a[i* nh + j] / sqrt(A_g[i][j]);
            }
        }
    }


    void train(int s, int r, int o, int ss, int rr, int oo) {
        vector<double> d_s;
        vector<double> d_r;
        vector<double> d_o;
        double* d_a = new double[nh*nh];//const_matrix(nh, nh, 0);

        vector<double> d_ss;
        vector<double> d_rr;
        vector<double> d_oo;
        double* d_aa = new double[nh*nh];

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

            if (ss != s)
                l2_normalize(E[ss]);

            if (oo != o)
                l2_normalize(E[oo]);

        }

        delete []d_a;
        delete []d_aa;
    }

};
#endif
