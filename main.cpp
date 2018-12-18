#include"models/DistMult.cpp"
#include"models/TransE.cpp"
#include"models/Analogy.cpp"
#include"models/ComplEx.cpp"

// based on Google's word2vec
int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    return -1;
}

int main(int argc, char **argv) {
    // option parser
    string  dataset     =  "FB15k/freebase_mtr100_mte100";
    string  algorithm   =  "Analogy";
    int     embed_dim   =  200;
    double  eta         =  0.1;
    double  gamma       =  1e-3;
    int     neg_ratio   =  6;
    int     num_epoch   =  500;
    int     num_thread  =  32;
    int     eval_freq   =  50;
    string  model_path;
    bool    prediction  = false;
    int     num_scalar  = 100;

    int i;
    if ((i = ArgPos((char *)"-algorithm",  argc, argv)) > 0)  algorithm   =  string(argv[i+1]);
    if ((i = ArgPos((char *)"-embed_dim",  argc, argv)) > 0)  embed_dim   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-eta",        argc, argv)) > 0)  eta         =  atof(argv[i+1]);
    if ((i = ArgPos((char *)"-gamma",      argc, argv)) > 0)  gamma       =  atof(argv[i+1]);
    if ((i = ArgPos((char *)"-neg_ratio",  argc, argv)) > 0)  neg_ratio   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-num_epoch",  argc, argv)) > 0)  num_epoch   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-num_thread", argc, argv)) > 0)  num_thread  =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-eval_freq",  argc, argv)) > 0)  eval_freq   =  atoi(argv[i+1]);
    if ((i = ArgPos((char *)"-model_path", argc, argv)) > 0)  model_path  =  string(argv[i+1]);
    if ((i = ArgPos((char *)"-dataset",    argc, argv)) > 0)  dataset     =  string(argv[i+1]);
    if ((i = ArgPos((char *)"-prediction", argc, argv)) > 0)  prediction  =  true;
    if ((i = ArgPos((char *)"-num_scalar", argc, argv)) > 0)  num_scalar  =  atoi(argv[i+1]);

    printf("dataset     =  %s\n", dataset.c_str());
    printf("algorithm   =  %s\n", algorithm.c_str());
    printf("embed_dim   =  %d\n", embed_dim);
    printf("eta         =  %e\n", eta);
    printf("gamma       =  %e\n", gamma);
    printf("neg_ratio   =  %d\n", neg_ratio);
    printf("num_epoch   =  %d\n", num_epoch);
    printf("num_thread  =  %d\n", num_thread);
    printf("eval_freq   =  %d\n", eval_freq);
    printf("model_path  =  %s\n", model_path.c_str());
    printf("num_scalar  =  %d\n", num_scalar);

    vector<string> ents = read_first_column(dataset + "-entities.txt");
    vector<string> rels = read_first_column(dataset + "-relations.txt");

    unordered_map<string, int> ent_map = create_id_mapping(ents);
    unordered_map<string, int> rel_map = create_id_mapping(rels);

    int ne = ent_map.size();
    int nr = rel_map.size();

    vector<triplet> sros_tr = create_sros(dataset + "-train.txt", ent_map, rel_map);
    vector<triplet> sros_va = create_sros(dataset + "-valid.txt", ent_map, rel_map);
    vector<triplet> sros_te = create_sros(dataset + "-test.txt",  ent_map, rel_map);
    vector<triplet> sros_al;

    sros_al.insert(sros_al.end(), sros_tr.begin(), sros_tr.end());
    sros_al.insert(sros_al.end(), sros_va.begin(), sros_va.end());
    sros_al.insert(sros_al.end(), sros_te.begin(), sros_te.end());

    SROBucket sro_bucket_al(sros_al);

    Model *model = NULL;

    if (algorithm == "DistMult")  model = new DistMult(ne, nr, embed_dim, eta, gamma);
    if (algorithm == "TransE")  model = new TransE(ne, nr, embed_dim, eta, gamma);
    if (algorithm == "ComplEx")  model = new ComplEx(ne, nr, embed_dim, eta, gamma);
    if (algorithm == "Analogy")  model = new Analogy(ne, nr, embed_dim, num_scalar, eta, gamma);


    assert(model != NULL);

    if (prediction) {
        Evaluator evaluator_te(ne, nr, sros_te, sro_bucket_al);
        model->load(model_path);
        auto info_te = evaluator_te.evaluate(model, -1);
        pretty_print("TE", info_te);
        return 0;
    }

    Evaluator evaluator_va(ne, nr, sros_va, sro_bucket_al);
    Evaluator evaluator_tr(ne, nr, sros_tr, sro_bucket_al);
    Evaluator evaluator_te(ne, nr, sros_te, sro_bucket_al);

    // thread-specific negative samplers
    vector<NegativeSampler> neg_samplers;
    for (int tid = 0; tid < num_thread; tid++)
        neg_samplers.push_back( NegativeSampler(ne, nr, rand() ^ tid) );

    int N = sros_tr.size();
    vector<int> pi = range(N);

    clock_t start;
    double elapse_tr = 0;
    double elapse_ev = 0;
    double best_mrr = 0;

    omp_set_num_threads(num_thread);
    for (int epoch = 1; epoch <= num_epoch; epoch++) {
        // evaluation
        if (epoch % eval_freq == 0) {
            start = omp_get_wtime();
            auto info_tr = evaluator_tr.evaluate(model, 2048);
            auto info_va = evaluator_va.evaluate(model, 2048);
            elapse_ev = omp_get_wtime() - start;

            // save the best model to disk
            double curr_mrr = (info_va["mrr_s"] + info_va["mrr_o"])/2;
            if (curr_mrr > best_mrr) {
                best_mrr = curr_mrr;
                if ( !model_path.empty() )
                    model->save(model_path);
            }

            printf("\n");
            printf("            EV Elapse    %f\n", elapse_ev);
            printf("======================================\n");
            pretty_print("TR", info_tr);
            printf("\n");
            pretty_print("VA", info_va);
            printf("\n");
            printf("VA  MRR_BEST    %.2f\n", 100*best_mrr);
            printf("\n");
        }

        shuffle(pi.begin(), pi.end(), GLOBAL_GENERATOR);

        start = omp_get_wtime();
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            triplet sro = sros_tr[pi[i]];
            int s = get<0>(sro);
            int r = get<1>(sro);
            int o = get<2>(sro);

            int tid = omp_get_thread_num();

            // positive example
            model->train(s, r, o, true);

            // negative examples
            for (int j = 0; j < neg_ratio; j++) {
                int oo = neg_samplers[tid].random_entity();
                int ss = neg_samplers[tid].random_entity();
                int rr = neg_samplers[tid].random_relation();

                // XXX: it is empirically beneficial to carry out updates even if oo == o || ss == s.
                // This might be related to regularization.
                model->train(s, r, oo, false);
                model->train(ss, r, o, false);
                model->train(s, rr, o, false);   // this improves MR slightly
            }
        }
        elapse_tr = omp_get_wtime() - start;
        printf("Epoch %03d   TR Elapse    %f\n", epoch, elapse_tr);
            
    }

    return 0;
}

