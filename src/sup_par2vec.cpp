#include "sup_par2vec.hpp"
#include "utils.hpp"
#include "measures.hpp"
SupPar2Vec::SupPar2Vec() :
        m_corpus_set(0),
        m_margin(1),
        m_alpha(1/3.0),
        m_beta(1/3.0),
        m_gamma(1/3.0),
        m_sample(0),
        m_word_max_norm(0),
        m_doc_max_norm(0),
        m_label_max_norm(0),
        m_lambda(0),
        m_num_iters(1),
        m_num_threads(1),
        m_loaded_params(false),
        m_is_pretrain(false),
        m_is_train_labeled_data(false),
        m_is_infer(false),
        m_vec_concat(0),
        m_pretrain_iters(0),
        m_verbose(0),
        m_model_save_path(""),
        num_processed(0),
        n_finished_(0),
        m_export_interm_model_interval(0),
        m_check_validation_err_interval(0),
        strand_(io),
        signals(io),
        timelimit_signals(io),
        model_save_timer(io),
        valid_check_timer(io),
        D0(nullptr)
{
}

SupPar2Vec::SupPar2Vec(
            long long wordvec_dim,
            int window_size,
            real learning_rate,
            int use_dm,
            int use_dbow,
            int use_hs,
            int negative,
            real margin,
            real alpha,
            real beta,
            real gamma,
            real sample,
            real word_max_norm,
            real doc_max_norm,
            real label_max_norm,
            real lambda,
            int num_iters,
            int export_interm_model_interval,
            int check_validation_err_interval,
            std::string model_save_path,
            int vec_concat,
            int pretrain_iters,
            int valid_test,
            int verbose)
      : m_hCorpus(nullptr),
        m_corpus_set(0),
        m_margin(margin),
        m_alpha(alpha),
        m_beta(beta),
        m_gamma(gamma),
        m_sample(sample),
        m_word_max_norm(word_max_norm),
        m_doc_max_norm(doc_max_norm),
        m_label_max_norm(label_max_norm),
        m_lambda(lambda),
        m_num_iters(num_iters),
        m_num_threads(1),
        m_loaded_params(false),
        m_is_pretrain(false),
        m_is_train_labeled_data(false),
        m_is_infer(false),
        m_vec_concat(vec_concat),
        m_pretrain_iters(pretrain_iters),
        m_valid_test(valid_test),
        m_verbose(verbose),
        m_model_save_path(model_save_path),
        num_processed(0),
        n_finished_(0),
        m_export_interm_model_interval(export_interm_model_interval),
        m_check_validation_err_interval(check_validation_err_interval),
        strand_(io),
        signals(io),
        timelimit_signals(io),
        model_save_timer(io),
        valid_check_timer(io),
        D0(nullptr)
{
  // Initialize parameters
  m_params.d = wordvec_dim;
  m_params.wn = window_size;
  m_params.hid_dim = vec_concat ? wordvec_dim*window_size : wordvec_dim;
  m_params.lr = learning_rate;
  m_params.dm = use_dm;
  m_params.dbow = use_dbow;
  m_params.hs = use_hs;
  m_params.margin = m_margin;
  m_params.alpha = m_alpha;
  m_params.beta = m_beta;
  m_params.gamma = m_gamma;
  m_params.sample = m_sample;
  m_params.word_max_norm = m_word_max_norm;
  m_params.doc_max_norm = m_doc_max_norm;
  m_params.label_max_norm = m_label_max_norm;
  m_params.lambda = m_lambda;
  m_params.vec_concat = m_vec_concat;
  m_params.negative = negative;
  m_params.num_iters = num_iters;
  m_params.pretrain_iters = m_pretrain_iters;
  m_params.valid_test = m_valid_test;

  CHECK(use_dm ^ use_dbow) << "Allows to select either DM or DBOW";
}

SupPar2Vec::~SupPar2Vec()
{
  delete m_table;
  DLOG(INFO) << "SupPar2Vec has been successfully destructed..";
}

void SupPar2Vec::setCorpus(HDF5Corpus *corpus)
{
  m_hCorpus = corpus;
  m_hCorpus->build_internals(true);

  m_params.V = m_hCorpus->get_vocabsize();
  m_params.trM = m_hCorpus->get_num_train_instances();
  m_params.vaM = m_hCorpus->get_num_valid_instances();
  m_params.tsM = m_hCorpus->get_num_test_instances();
  m_params.L = m_hCorpus->get_num_seen_labels();

  if(m_params.L > 0) 
  {
    m_is_train_labeled_data = true;
  }
  else {
    // set all configurations related to labels 
    m_params.alpha = 0;   
    m_params.alpha = 0;   
    float sum = m_params.beta + m_params.gamma;
    m_params.beta = m_params.beta/sum;   
    m_params.gamma = m_params.beta/sum;   
    m_params.margin = 0;
    m_params.label_max_norm = 0;
    m_params.lambda = 0;

    m_params.pretrain_iters = 0;
  }
}

void SupPar2Vec::run_infer_unseen_documents(int thread_id, real *uD0)
{
  boost::random_device rd; 
  boost::random::mt19937 rng(rd()); 
  UniformRandom unif_rand(rng, 0.0, 1.0);

  long long num_unseen_instances = m_hCorpus->get_num_test_instances();

  const long long max_instances = (long long) ceil(num_unseen_instances/(real)m_num_threads);

  // local variables per thread
  std::unique_ptr<real[]> hid(new real[m_params.hid_dim]());
  std::unique_ptr<real[]> grad(new real[m_params.hid_dim]());

  real lr = m_params.lr;

  int base_instance_idx = max_instances*thread_id;
  std::set<int> label_set;
  int last_idx = std::min(num_unseen_instances, base_instance_idx+max_instances);
  const int mb_sz = 10000;

  std::vector<int> sampled_doc_word_seq(0);

  long long paragraph_idx = 0;

  for(int iter = 0; iter < m_params.num_iters; ++iter)
  {
    for(int stx_pos=base_instance_idx; stx_pos < last_idx; stx_pos += mb_sz)
    {
      int num_instances = (stx_pos + mb_sz >= last_idx)?(last_idx-stx_pos):mb_sz;

      std::vector<std::vector<int>> words_in_docs(num_instances, std::vector<int>(0));    // each vector contains a word sequence for a document
      std::vector<std::vector<int>> labels_in_docs(num_instances, std::vector<int>(0));   // a set of labels

      m_corpus_set[thread_id]->getTestInstances(stx_pos, num_instances, words_in_docs, labels_in_docs);

      for(int inst_idx=0; inst_idx < num_instances; inst_idx++)
      {
        memset(hid.get(), 0, m_params.hid_dim*sizeof(real));
        memset(grad.get(), 0, m_params.hid_dim*sizeof(real));

        paragraph_idx = stx_pos+inst_idx;
        std::vector<int>& doc_word_seq = words_in_docs[inst_idx];
        sample_words(doc_word_seq, sampled_doc_word_seq, m_params.sample, unif_rand);    // sample out words by frequency

        real *x = uD0+paragraph_idx*m_params.d;
        distributed_memory(lr*m_params.beta, m_params.word_max_norm, m_params.doc_max_norm, rng, sampled_doc_word_seq, x, hid.get(), grad.get(), true);
      }
    }
  }
}

real* SupPar2Vec::infer_unseen_documents()
{
  openblas_set_num_threads(1);

  long long num_unseen_documents = m_hCorpus->get_num_test_instances();

  boost::random_device rd; 
  boost::random::mt19937 rng(rd()); 

  UniformRandom unif_rand(rng, 0.0, 1.0);
  NormalRandom normal_rand(rng, 0.0, 1.0);

  real *uD0 = new real[m_params.d * num_unseen_documents];
  CHECK(uD0) << "Memory allocation failed: " 
            << m_params.d << " x " 
            << num_unseen_documents << " (" 
            << (m_params.d*num_unseen_documents*sizeof(real)/(real)1000000) << " MB)";

  long long a,b;
  for(a = 0; a < num_unseen_documents; a++)
  {
    for(b = 0; b < m_params.d; b++)
    {
      uD0[b+a*m_params.d] = normal_rand()/sqrt(m_params.trM);
    }
  }
  m_corpus_set.clear();
  for(int i=0; i < m_num_threads; i++)
  {
    m_corpus_set.push_back(std::unique_ptr<HDF5Corpus>(new HDF5Corpus(m_hCorpus->getDatasetFilename())));
  }
  for(int i=0; i < m_num_threads; i++)
  {
    m_corpus_set[i]->build_internals(false);
  }

  LOG(INFO) << m_params;

  boost::thread_group threads;
  for (int i = 0; i < m_num_threads; i++)
  {
    threads.create_thread(boost::bind(&SupPar2Vec::run_infer_unseen_documents, this, i, uD0));
  }
  threads.join_all();

  return uD0;
}

void SupPar2Vec::run_infer_unseen_labels(int thread_id, real *uW0)
{
  boost::random_device rd; 
  boost::random::mt19937 rng(rd()); 
  UniformRandom unif_rand(rng, 0.0, 1.0);

  long long num_seen_labels = m_params.L;
  long long num_unseen_labels = m_hCorpus->get_num_labels() - num_seen_labels;

  const long long max_labels = (long long) ceil(num_unseen_labels/(real)m_num_threads);

  // local variables per thread
  std::unique_ptr<real[]> hid(new real[m_params.hid_dim]());
  std::unique_ptr<real[]> grad(new real[m_params.hid_dim]());

  std::vector<int> label_word_seq(0);
  std::vector<int> sampled_label_word_seq(0);

  real lr = m_params.lr;

  int base_label_idx = max_labels*thread_id;
  int last_idx = std::min(num_unseen_labels, base_label_idx+max_labels);
  int mb_sz = last_idx - base_label_idx;

  for(int iter = 0; iter < m_params.num_iters; ++iter)
  {
    for(int i=0; i < mb_sz; i++)
    {
      m_hCorpus->getWordSeqFromLabelDesc(num_seen_labels+base_label_idx+i, label_word_seq);
      sample_words(label_word_seq, sampled_label_word_seq, m_params.sample, unif_rand);    // sample out words by frequency

      real *y = uW0+(base_label_idx+i)*m_params.d;
      distributed_memory(lr*m_params.gamma, m_params.word_max_norm, m_params.label_max_norm, rng, sampled_label_word_seq, y, hid.get(), grad.get(), true);
    }
  }

}

real* SupPar2Vec::infer_unseen_labels()
{
  openblas_set_num_threads(1);

  long long num_seen_labels = m_params.L;
  long long num_unseen_labels = m_hCorpus->get_num_labels() - num_seen_labels;

  if(num_unseen_labels == 0) return NULL;

  boost::random_device rd; 
  boost::random::mt19937 rng(rd()); 

  UniformRandom unif_rand(rng, 0.0, 1.0);
  NormalRandom normal_rand(rng, 0.0, 1.0);

  real *uW0 = new real[m_params.d * num_unseen_labels];
  CHECK(uW0) << "Memory allocation failed: " 
            << m_params.d << " x " 
            << num_unseen_labels << " (" 
            << (m_params.d*num_unseen_labels*sizeof(real)/(real)1000000) << " MB)";

  long long a,b;
  for(a = 0; a < num_unseen_labels; a++)
  {
    for(b = 0; b < m_params.d; b++)
    {
      uW0[b+a*m_params.d] = normal_rand()/sqrt(num_seen_labels);
    }
  }

  LOG(INFO) << m_params;

  boost::thread_group threads;
  for (int i = 0; i < m_num_threads; i++)
  {
    threads.create_thread(boost::bind(&SupPar2Vec::run_infer_unseen_labels, this, i, uW0));
  }
  threads.join_all();

  return uW0;
}

void SupPar2Vec::compute_validation_cost()
{
  unsigned int base_idx = 0;
  long long num_instances = 0;
  if(m_params.valid_test)
    num_instances = m_hCorpus->get_num_test_instances();
  else
    num_instances = m_hCorpus->get_num_valid_instances();

  num_instances = 10000;

  int L = m_params.L;

  boost::random_device rd;
  boost::mt19937 rng(rd());

  UniformRandom unif_rand(rng, 0.0, 1.0);
  NormalRandom normal_rand(rng, 0.0, 1.0);
  IntRandom int_rand(rng, 0, m_params.trM-num_instances-1);

  real *vaD0 = new real[m_params.d * num_instances];
  CHECK(vaD0) << "Memory allocation failed: " 
            << m_params.d << " x " 
            << num_instances << " (" 
            << (m_params.d*num_instances*sizeof(real)/(real)1000000) << " MB)";

  long long a,b;
  for(a = 0; a < num_instances; a++)
  {
    for(b = 0; b < m_params.d; b++)
    {
      vaD0[b+a*m_params.d] = normal_rand()/sqrt(m_params.trM);
    }
  }
  
  std::unique_ptr<real[]> hid(new real[m_params.hid_dim]());
  std::unique_ptr<real[]> grad(new real[m_params.hid_dim]());
  std::unique_ptr<real[]> scores(new real[m_params.L]());

  std::vector<std::vector<int>> words_in_docs(num_instances,std::vector<int>(0)), labels_in_docs(num_instances,std::vector<int>(0));
  std::vector<int> sampled_doc_word_seq(0);

  real lr = m_params.lr;
  if(m_params.valid_test)
  {
    IntRandom test_base_idx_rand(rng, 0, m_params.tsM-num_instances-1);
    base_idx = test_base_idx_rand();
    m_hCorpus->getTestInstances(base_idx, num_instances, words_in_docs, labels_in_docs);
    for(size_t i=0; i < labels_in_docs.size(); i++)
    {
      std::vector<int>& labels = labels_in_docs[i];
      labels.erase(std::remove_if(labels.begin(), labels.end(), [&L](int x){return x >= L;}), labels.end());  // erase label indices if they are greater than the biggest label index in the training set
    }
  }
  else
  {
    IntRandom valid_base_idx_rand(rng, 0, m_params.vaM-num_instances-1);
    base_idx = valid_base_idx_rand();
    m_hCorpus->getValidInstances(base_idx, num_instances, words_in_docs, labels_in_docs);
  }


  for(int iter = 0; iter < (m_params.sentences_seen_actual/(real)m_params.trM) + m_pretrain_iters + 1; ++iter)
  {
    for(int doc_idx = 0; doc_idx < num_instances; doc_idx++)
    {
      std::vector<int>& doc_word_seq = words_in_docs[doc_idx];
      sample_words(doc_word_seq, sampled_doc_word_seq, m_params.sample, unif_rand);    // sample out words by frequency

      real *x = vaD0+doc_idx*m_params.d;
      distributed_memory(lr*m_params.beta, m_params.word_max_norm, m_params.doc_max_norm, rng, sampled_doc_word_seq, x, hid.get(), grad.get(), true);
    }
  }

  int lda = m_params.d;
  int ldb = m_params.d;
  int ldc = 1;

  std::set<int> label_set;

  real avg_rankloss = 0.;
  real loss=0.;
  for(long long doc_idx = 0; doc_idx < num_instances; doc_idx++)
  {
    std::vector<int>& labels = labels_in_docs[doc_idx];

    label_set = std::set<int>(labels.begin(), labels.end());

    real *x = vaD0+doc_idx*m_params.d;
    //cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, m_params.L, m_params.d, 1.0, x,lda,W0.get(),ldb,0.0,scores,ldc);
    cblas_sgemv(CblasColMajor, CblasNoTrans, m_params.d, m_params.d, 1.0, V.get(), m_params.d, x, 1, 0.0, hid.get(), 1);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, m_params.L, m_params.d, 1.0, hid.get(),lda,W0.get(),ldb,0.0,scores.get(),ldc);
    loss = Measures::rankloss(label_set, scores.get(), m_params.L);
    avg_rankloss += loss;
  }

  real avg_train_rankloss = 0.;
  real *D0_(nullptr);
  base_idx = int_rand();
  m_hCorpus->getTrainInstances(base_idx, num_instances, words_in_docs, labels_in_docs);
  D0_ = D0.get();
  for(long long doc_idx = 0; doc_idx < num_instances; doc_idx++)
  {
    std::vector<int>& labels = labels_in_docs[doc_idx];
    label_set = std::set<int>(labels.begin(), labels.end());

    real *x = D0_+(base_idx+doc_idx)*m_params.d;
    //cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, m_params.L, m_params.d, 1.0, x,lda,W0.get(),ldb,0.0,scores,ldc);
    cblas_sgemv(CblasColMajor, CblasNoTrans, m_params.d, m_params.d, 1.0, V.get(), m_params.d, x, 1, 0.0, hid.get(), 1);
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, 1, m_params.L, m_params.d, 1.0, hid.get(),lda,W0.get(),ldb,0.0,scores.get(),ldc);
    avg_train_rankloss += Measures::rankloss(label_set, scores.get(), m_params.L);
  }

  LOG(INFO) << "After iterating the dataset " << std::fixed << std::setprecision(3) << m_params.sentences_seen_actual / (real) m_params.trM  << " times:\t" 
            << std::setprecision(6) 
            << avg_train_rankloss/(real)num_instances << "\t"
            << avg_rankloss/(real)num_instances << "\n\n";

  delete[] vaD0;

  if(n_finished_ == m_num_threads)
  {
    model_save_timer.cancel();
    valid_check_timer.cancel();
    std::raise(SIGINT);
  }
}

void SupPar2Vec::save_model()
{
  LOG(INFO) << "Save to: " << m_model_save_path;
  if(m_model_save_path.size() > 0)
  {
    std::ofstream out_stream(m_model_save_path);
    boost::archive::binary_oarchive oar(out_stream);
    oar << *this;
    out_stream.close();
    LOG(INFO) << "Intermediate model saved";

    export_vectors(m_model_save_path + ".vectors");
  }
}

void SupPar2Vec::interrupt_handler(
    const boost::system::error_code& error,
    int signal_number)
{
  if(n_finished_ < m_num_threads)
  {
    if (!error)
    {
      static boost::atomic_bool first(true);
      if(first) {
          std::cout << " A signal(SIGINT) occurred." << std::endl;
          first = false;

          save_model();

          signals.async_wait(boost::bind(&SupPar2Vec::interrupt_handler, this, _1, _2));
      }
      else {
          std::cout << " A second signal(SIGINT) occurred, exiting...." << std::endl;
          exit(1);
      }
    }
  }
}

void SupPar2Vec::timelimit_handler(
    const boost::system::error_code& error,
    int signal_number)
{
  if (!error)
  {
    save_model();
    std::raise(SIGTERM);
  }
}

void SupPar2Vec::check_save_time(const boost::system::error_code& ec)
{
  if(n_finished_ < m_num_threads)
  {
    save_model();

    model_save_timer.expires_at(model_save_timer.expires_at() + boost::posix_time::minutes(m_export_interm_model_interval));
    model_save_timer.async_wait(strand_.wrap(boost::bind(&SupPar2Vec::check_save_time, this, _1)));
  }
}

void SupPar2Vec::check_validation_time(const boost::system::error_code& ec)
{
  if(n_finished_ < m_num_threads)
  {
    compute_validation_cost();

    valid_check_timer.expires_at(valid_check_timer.expires_at() + boost::posix_time::minutes(m_check_validation_err_interval));
    valid_check_timer.async_wait(strand_.wrap(boost::bind(&SupPar2Vec::check_validation_time, this, _1)));
  }
}

void SupPar2Vec::normalize(real *vec, real max_norm)
{
  if(max_norm > 0)
  {
    real norm = cblas_snrm2(m_params.d, vec, 1);
    if(norm > max_norm) cblas_sscal(m_params.d, max_norm/norm, vec, 1);
  }
}

void SupPar2Vec::pretrain(int num_threads)
{
  if(m_is_train_labeled_data)
  {
    int num_iters = m_params.num_iters;
    float alpha = m_params.alpha;
    m_params.num_iters = m_pretrain_iters;
    m_params.alpha = 0;

    m_is_pretrain = true;
    this->train(num_threads);
    m_is_pretrain = false;
    n_finished_ = 0;
    m_params.sentences_seen_actual = 0;

    m_params.num_iters = num_iters;
    m_params.alpha = alpha;

    LOG(INFO) << "Document and word representations have been trained.";
  }
  else  m_params.pretrain_iters = 0;
}

void SupPar2Vec::train(int num_threads)
{
  openblas_set_num_threads(1);

  m_num_threads = num_threads;
  LOG(INFO) << this->getParams();

  //LOG(INFO) << "Create multiple datasets";
  for(int i=0; i < m_num_threads; i++)
  {
    m_corpus_set.push_back(std::unique_ptr<HDF5Corpus>(new HDF5Corpus(m_hCorpus->getDatasetFilename())));
  }
  for(int i=0; i < m_num_threads; i++)
  {
    m_corpus_set[i]->build_internals(false);
  }

  m_start_time = std::chrono::system_clock::now();

  if(!m_is_pretrain && !m_is_infer)
  {
    signals.add(SIGINT);
    timelimit_signals.add(SIGUSR2);

    if(m_export_interm_model_interval > 0)
    {
      model_save_timer.expires_from_now(boost::posix_time::minutes(m_export_interm_model_interval));
      model_save_timer.async_wait(strand_.wrap(boost::bind(&SupPar2Vec::check_save_time, this, _1)));
    }

    if(m_check_validation_err_interval > 0 && m_is_train_labeled_data)
    {
      valid_check_timer.expires_from_now(boost::posix_time::minutes(m_check_validation_err_interval));
      valid_check_timer.async_wait(strand_.wrap(boost::bind(&SupPar2Vec::check_validation_time, this, _1)));
    }

    signals.async_wait(boost::bind(&SupPar2Vec::interrupt_handler, this, _1, _2));
    timelimit_signals.async_wait(boost::bind(&SupPar2Vec::timelimit_handler, this, _1, _2));
  }

  boost::thread_group threads;
  for (int i = 0; i < m_num_threads; i++)
  {
    threads.create_thread(boost::bind(&SupPar2Vec::run_train, this, i));
  }

  if(!m_is_pretrain && !m_is_infer)
  {
    threads.create_thread(boost::bind(&boost::asio::io_service::run, &io));
    io.run();

    save_model();
  }

  threads.join_all();

  signals.clear();
  timelimit_signals.clear();
}

void SupPar2Vec::infer(int num_iters, int num_threads, int verbose, std::string output_path)
{
  boost::random_device rd;
  boost::random::mt19937 rng(rd());
  NormalRandom normal_rand(rng, 0.0, 1.0);

  long long num_unseen_documents = m_hCorpus->get_num_test_instances();
  //real train_lr;
  int train_num_iters, train_num_threads, train_verbose;
  long long train_sentences_seen_actual;

  m_is_infer = true;
  num_processed = 0;

  // store parameters used during training into temproray variables

  train_num_iters = m_params.num_iters;
  if(num_iters > 0)
  {
    m_params.num_iters = num_iters;
  }
  else
  {
    m_params.num_iters = m_params.sentences_seen_actual/(real)m_params.trM + 1;
  }

  train_sentences_seen_actual = m_params.sentences_seen_actual;
  m_params.sentences_seen_actual = 0;

  train_num_threads = m_num_threads;
  m_num_threads = num_threads;

  train_verbose = m_verbose;
  m_verbose = verbose;

  LOG(INFO) << "Started inference on the test data";

  real *uD0 = infer_unseen_documents();

  LOG(INFO) << "Completed inference on the test data";

  m_params.sentences_seen_actual = train_sentences_seen_actual;
  m_params.num_iters = train_num_iters;
  m_num_threads = train_num_threads;
  m_verbose = train_verbose;
  m_is_infer = false;

  LOG(INFO) << "Started to infer unseen labels";
  m_num_threads = num_threads;
  real *uW0 = infer_unseen_labels();

  std::ofstream file(output_path, std::ios::out | std::ios::binary);

  CHECK(file.is_open()) << "Failed to open the file: " << output_path;

  file.write((char *) &m_params.d, sizeof(long long));
  file.write((char *) &num_unseen_documents, sizeof(long long));
  file.write((char *) uD0, m_params.d*num_unseen_documents*sizeof(real));

  delete uD0;

  if(uW0)
  {
    long long num_unseen_labels = m_hCorpus->get_num_labels() - m_params.L;

    file.write((char *) &m_params.d, sizeof(long long));
    file.write((char *) &num_unseen_labels, sizeof(long long));
    file.write((char *) uW0, m_params.d*num_unseen_labels*sizeof(real));

    delete uW0;
  }

  file.close();
}

void SupPar2Vec::init()
{
  if(!m_loaded_params)
  {
    m_params.sentences_seen_actual = 0;

    LOG(INFO) << "The number of training instances in the corpus: " << m_params.trM;
    LOG(INFO) << "The number of words in vocabulary: " << m_params.V;
    LOG(INFO) << "The number of seen labels: " << m_params.L;
    long long a, b;

    LOG(INFO) << "Initializing the model";

    boost::random_device rd;
    boost::random::mt19937 rng(rd());
    NormalRandom normal_rand(rng, 0.0, 1.0);

    real *D0_ = new real[m_params.d * m_params.trM];
    CHECK(D0_) << "Memory allocation failed: " 
              << m_params.d << " x " 
              << m_params.trM << " (" 
              << (m_params.d*m_params.trM*sizeof(real)/(real)1000000) << " MB)";

    for(a = 0; a < m_params.trM; a++)
    {
      for(b = 0; b < m_params.d; b++)
      {
        D0_[b+a*m_params.d] = normal_rand()/sqrt(m_params.trM);
      }
    }
    D0.reset(D0_);

    real *U0_ = new real [m_params.d * (m_params.V+2)];
    CHECK(U0_) << "Memory allocation failed: " 
              << m_params.d << " x " 
              << (m_params.V+2) << " (" 
              << (m_params.d*(m_params.V+2)*sizeof(real)/(real)1000000) << " MB)";

    for(a = 0; a < m_params.V+2; a++)
    {
      for(b = 0; b < m_params.d; b++)
      {
        U0_[b+a*m_params.d] = normal_rand()/sqrt(m_params.V);
      }
    }
    U0.reset(U0_);

    if(m_is_train_labeled_data)
    {
      real *V_ = new real[m_params.d * m_params.d];
      CHECK(V_) << "Memory allocation failed: " 
                << m_params.d << " x " 
                << m_params.d << " (" 
                << (m_params.d*m_params.d*sizeof(real)/(real)1000000) << " MB)";

      for(a = 0; a < m_params.d; a++)
      {
        for(b = 0; b < m_params.d; b++)
        {
          V_[b+a*m_params.d] = normal_rand()/sqrt(m_params.d);
        }
      }
      V.reset(V_);

      real *W0_ = new real[m_params.d * m_params.L];
      CHECK(W0_) << "Memory allocation failed: " 
                << m_params.d << " x " 
                << m_params.L << " (" 
                << (m_params.d*m_params.L*sizeof(real)/(real)1000000) << " MB)";

      for(a = 0; a < m_params.L; a++)
      {
        for(b = 0; b < m_params.d; b++)
        {
          W0_[b+a*m_params.d] = normal_rand()/sqrt(m_params.L);
        }
      }
      W0.reset(W0_);
    }

    if(m_params.hs)
    {
      real *U1_ = new real [m_params.hid_dim*m_params.V];
      CHECK(U1_) << "Memory allocation failed: " 
                << m_params.hid_dim << " x " 
                << m_params.V << " (" 
                << (m_params.hid_dim*m_params.V*sizeof(real)/(real)1000000) << " MB)";

      for(a = 0; a < m_params.V; a++)
      {
        for(b = 0; b < m_params.hid_dim; b++)
        {
          U1_[b+a*m_params.hid_dim] = normal_rand()/sqrt(m_params.V);
        }
      }
      U1.reset(U1_);
    }

    if(m_params.negative > 0)
    {
      real *U2_ = new real [m_params.hid_dim*m_params.V];
      CHECK(U2_) << "Memory allocation failed: " 
                << m_params.hid_dim << " x " 
                << m_params.V << " (" 
                << (m_params.hid_dim*m_params.V*sizeof(real)/(real)1000000) << " MB)";

      for(a = 0; a < m_params.V; a++)
      {
        for(b = 0; b < m_params.hid_dim; b++)
        {
          U2_[b+a*m_params.hid_dim] = normal_rand()/sqrt(m_params.V);
        }
      }
      U2.reset(U2_);
    }
  }

  createTable();
}

inline real sigmoid(real x)
{
  return 1/(1+exp(-x));
}

void SupPar2Vec::distributed_bow(real lr, real word_max_norm, real entity_max_norm, boost::mt19937 &rng, std::vector<int> &tokens, real *entity_vec, real *grad, const bool is_infer)
{
  std::vector<vocab_ptr>& vocabulary = m_hCorpus->getVocabulary();

  real *U1_(nullptr), *U2_(nullptr);

  if(m_params.hs)           U1_ = U1.get();
  if(m_params.negative > 0) U2_ = U2.get();

  for(int pos=0; pos < (int) tokens.size(); ++pos)
  {
    long long output_word_idx = -1;

    for(int j=pos-(m_params.wn)/2.0; j < pos+(m_params.wn)/2.0+1; ++j)
    {
      if (j == pos) continue;
      if (j < 0 || j >= (int) tokens.size()) continue;
      output_word_idx = tokens[j];
      memset(grad, 0, m_params.d*sizeof(real));
      if (m_params.hs)
      {
        for(int k = 0; k < vocabulary[output_word_idx]->get_codelen(); ++k)
        {
          long long inner_node_idx = vocabulary[output_word_idx]->get_inner_node_idxAt(k);
          real f = sigmoid(cblas_sdot(m_params.d, 
                                      entity_vec, 1,
                                      U1_+inner_node_idx*m_params.d, 1)
                          ); 
          CHECK(!isnan(f) && !isinf(f));

          int code = vocabulary[output_word_idx]->get_codeAt(k);
          real delta = (1 - code - f) * lr;
          cblas_saxpy(m_params.d, delta, U1_+inner_node_idx*m_params.d, 1, grad, 1);
          if(!is_infer)
          {
            cblas_saxpy(m_params.d, delta, entity_vec, 1, U1_+inner_node_idx*m_params.d, 1);
            normalize(U1_+inner_node_idx*m_params.d, word_max_norm);
          }
        }
      }
      if (m_params.negative > 0)
      {
        int label;
        long long target_word_idx;
        for(int k = 0; k < m_params.negative + 1; ++k)
        {
          if (k == 0)
          {
            target_word_idx = output_word_idx;
            label = 1;
          }
          else
          {
            target_word_idx = m_table[rng() % table_size];
            if (target_word_idx == 0) target_word_idx = rng() % (m_params.V - 1) + 1;
            label = 0;
          }
          real f = sigmoid(cblas_sdot(m_params.d, 
                                      entity_vec, 1,
                                      U2_+target_word_idx*m_params.d, 1)
                          ); 
          CHECK(!isnan(f) && !isinf(f));

          real delta = (label - f) * lr;
          cblas_saxpy(m_params.d, delta, U2_+target_word_idx*m_params.d, 1, grad, 1);
          if(!is_infer)
          {
            cblas_saxpy(m_params.d, delta, entity_vec, 1, U2_+target_word_idx*m_params.d, 1);
            normalize(U2_+target_word_idx*m_params.d, word_max_norm);
          }
        }
      }
      cblas_saxpy(m_params.d, 1, grad, 1, entity_vec, 1);
      normalize(entity_vec, entity_max_norm);
    }
  }
}

void SupPar2Vec::distributed_memory(real lr, real word_max_norm, real entity_max_norm, boost::mt19937 &rng, std::vector<int> &tokens, real *entity_vec, real *hid, real *grad, const bool is_infer)
{
  std::vector<vocab_ptr>& vocabulary = m_hCorpus->getVocabulary();

  real *U0_(nullptr), *U1_(nullptr), *U2_(nullptr);

  U0_ = U0.get();
  if(m_params.hs)           U1_ = U1.get();
  if(m_params.negative > 0) U2_ = U2.get();

  for(int pos=0; pos < (int) tokens.size(); ++pos)
  {
    cblas_scopy(m_params.d, entity_vec, 1, hid, 1);
    long long output_word_idx = tokens[pos];
    long long input_word_idx;
    int num_components = 1;
    int component_pos = 0;

    for(int j=pos-(m_params.wn)/2; j < pos+(m_params.wn)/2+1; ++j)
    {
      if (j == pos) continue;
      if (j < 0)
      {
        if(!m_params.vec_concat) continue;
        input_word_idx = m_params.V;  // left padding
      }
      else if (j >= (int) tokens.size())
      {
        if(!m_params.vec_concat) continue;
        input_word_idx = m_params.V+1;  // right padding
      }
      else
      {
        input_word_idx = tokens[j];
      }
      if(m_params.vec_concat)  cblas_scopy(m_params.d, U0_+input_word_idx*m_params.d, 1, hid+(++component_pos)*m_params.d, 1);
      else            cblas_saxpy(m_params.d, 1, U0_+input_word_idx*m_params.d, 1, hid, 1);
      num_components++;
    }
    if(!m_params.vec_concat) cblas_sscal(m_params.hid_dim, 1/(real)num_components, hid, 1);
    memset(grad, 0, m_params.hid_dim*sizeof(real));
    if (m_params.hs)
    {
      for(int k = 0; k < vocabulary[output_word_idx]->get_codelen(); ++k)
      {
        long long inner_node_idx = vocabulary[output_word_idx]->get_inner_node_idxAt(k);
        real f = sigmoid(cblas_sdot(m_params.hid_dim, 
                                    hid, 1,
                                    U1_+inner_node_idx*m_params.hid_dim, 1)
                        ); 
        CHECK(!isnan(f) && !isinf(f));

        int code = vocabulary[output_word_idx]->get_codeAt(k);
        real delta = (1 - code - f) * lr;
        cblas_saxpy(m_params.hid_dim, delta, U1_+inner_node_idx*m_params.hid_dim, 1, grad, 1);
        if(!is_infer)
        {
          cblas_saxpy(m_params.hid_dim, delta, hid, 1, U1_+inner_node_idx*m_params.hid_dim, 1);
          normalize(U1_+inner_node_idx*m_params.hid_dim, word_max_norm);
        }
      }
    }
    if (m_params.negative > 0)
    {
      int label;
      long long target_word_idx;
      for(int k = 0; k < m_params.negative + 1; ++k)
      {
        if (k == 0)
        {
          target_word_idx = output_word_idx;
          label = 1;
        }
        else
        {
          target_word_idx = m_table[rng() % table_size];
          if (target_word_idx == 0) target_word_idx = rng() % (m_params.V - 1) + 1;
          if (target_word_idx == output_word_idx) continue;
          label = 0;
        }
        real f = sigmoid(cblas_sdot(m_params.hid_dim, 
                                    hid, 1,
                                    U2_+target_word_idx*m_params.hid_dim, 1)
                        ); 
        CHECK(!isnan(f) && !isinf(f));

        real delta = (label - f) * lr;

        // update
        cblas_saxpy(m_params.hid_dim, delta, U2_+target_word_idx*m_params.hid_dim, 1, grad, 1);
        if(!is_infer)
        {
          cblas_saxpy(m_params.hid_dim, delta, hid, 1, U2_+target_word_idx*m_params.hid_dim, 1);
          normalize(U2_+target_word_idx*m_params.hid_dim, word_max_norm);
        }
      }
    }

    if(!is_infer)
    {
      component_pos = 0;
      for(int j=pos-(m_params.wn)/2; j < pos+(m_params.wn)/2+1; ++j)
      {
        if (j == pos) continue;
        if (j < 0)
        {
          if(!m_params.vec_concat) continue;
          input_word_idx = m_params.V;  // left padding
        }
        else if (j >= (int) tokens.size())
        {
          if(!m_params.vec_concat) continue;
          input_word_idx = m_params.V+1;  // right padding
        }
        else
        {
          input_word_idx = tokens[j];
        }
        if(m_params.vec_concat)  cblas_saxpy(m_params.d, 1, grad+(++component_pos)*m_params.d, 1, U0_+input_word_idx*m_params.d, 1);
        else            cblas_saxpy(m_params.d, 1/(real)num_components, grad, 1, U0_+input_word_idx*m_params.d, 1);
        normalize(U0_+input_word_idx*m_params.d, word_max_norm);
      }
    }
    if(m_params.vec_concat)  cblas_saxpy(m_params.d, 1, grad, 1, entity_vec, 1);
    else            cblas_saxpy(m_params.d, 1/(real)num_components, grad, 1, entity_vec, 1);
    normalize(entity_vec, entity_max_norm);
  }
}

real SupPar2Vec::compute_warp_loss(IntRandom &int_rand, real lr, real doc_max_norm, real label_max_norm, std::vector<long long>& random_label_indices, long long paragraph_idx, std::vector<int> &label_indices, real *hid, real *grad, std::vector<long long>& relevant_labels, std::vector<long long> &irrelevant_labels)
{
  if(!m_is_train_labeled_data) return 0;

  relevant_labels.clear();
  irrelevant_labels.clear();

  real *D0_, *W0_;
  real *V_;

  D0_ = D0.get();
  W0_ = W0.get();
  V_ = V.get();
  
  real *x = D0_+paragraph_idx*m_params.d;
  std::set<long long> pos_label_set(label_indices.begin(), label_indices.end());
  real avg_loss = 0.;
  for(const long long &label_idx : label_indices)
  {
    // compute the similarity score between a paragraph vector and its relevant label vector
    cblas_sgemv(CblasColMajor, CblasNoTrans, m_params.d, m_params.d, 1.0, V_, m_params.d, x, 1, 0.0, hid, 1);
    real pos_score = cblas_sdot(m_params.d, hid, 1, W0_+label_idx*m_params.d, 1);
    CHECK(!isnan(pos_score) && !isinf(pos_score));
    
    //long long trials=0;
    //for(const long long& random_label_idx : random_label_indices)
    for(size_t trials=1; trials <= m_params.L-pos_label_set.size(); trials++)
    {
      long long random_label_idx = int_rand(); 
      const bool is_pos_label = pos_label_set.find(random_label_idx) != pos_label_set.end();

      if(!is_pos_label)
      {
        //trials++;
        // compute the similarity score between a paragraph vector and a random label vector
        real neg_score = cblas_sdot(m_params.d, hid, 1, W0_+random_label_idx*m_params.d, 1);
        CHECK(!isnan(neg_score) && !isinf(neg_score));

        real loss = m_params.margin-pos_score+neg_score;
        if( loss > 0)
        {
          avg_loss += loss/(real)label_indices.size();
          unsigned int expected_rank = std::floor((m_params.L - label_indices.size())/(real)trials);
          CHECK(expected_rank >= 1) << "Actual expected rank: " << label_indices.size() << " : " << expected_rank << " by " << trials;
          real warp_loss=0.;
          for(size_t k=0; k < expected_rank; k++) warp_loss += 1/(real)(k+1);

          real update_step = (lr*warp_loss)/(real)label_indices.size();
          // update paragraph and label vectors

          cblas_scopy(m_params.d, W0_+random_label_idx*m_params.d, 1, grad, 1);
          cblas_saxpy(m_params.d, -1, W0_+label_idx*m_params.d, 1, grad, 1);

          // update label embeddings
          cblas_saxpy(m_params.d, update_step, hid, 1, W0_+label_idx*m_params.d, 1);
          cblas_saxpy(m_params.d, -update_step, hid, 1, W0_+random_label_idx*m_params.d, 1);
          normalize(W0_+label_idx*m_params.d, m_params.label_max_norm);
          normalize(W0_+random_label_idx*m_params.d, m_params.label_max_norm);

          cblas_scopy(m_params.d, x, 1, hid, 1);
          // update doc vector
          cblas_sgemv(CblasColMajor,CblasTrans,m_params.d,m_params.d,-update_step,V_,m_params.d,grad,1,1.0,x,1);
          normalize(x, m_params.doc_max_norm);

          int lda = m_params.d;
          int ldb = m_params.d;
          int ldc = m_params.d;
          // update projection matrix
          cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,m_params.d,m_params.d,1,-update_step,grad,lda,hid,ldb,1.0,V_,ldc);
          cblas_saxpy(m_params.d*m_params.d,-lr*m_params.lambda,V_,1,V_,1);

          relevant_labels.push_back(label_idx);
          irrelevant_labels.push_back(random_label_idx);
          break;
        }
      }
    }
  }

  return avg_loss;
}

void SupPar2Vec::sample_words(std::vector<int>& orig, std::vector<int>& sampled, real sample_prob, UniformRandom &unif_rand)
{
  std::vector<vocab_ptr>& vocabulary = m_hCorpus->getVocabulary();
  long long word_idx;
  real prob, rand_num;
  sampled.clear();
  if (sample_prob > 0)
  {
    for(size_t i=0; i < orig.size(); i++)
    {
      // word index
      word_idx = orig[i];
      prob = (sqrt(vocabulary[word_idx]->m_freq / (sample_prob * m_hCorpus->get_train_words())) + 1) 
                  * (sample_prob * m_hCorpus->get_train_words()) / vocabulary[word_idx]->m_freq;
      rand_num = unif_rand();
      CHECK(rand_num >=0 && rand_num <= 1);
      if(prob < rand_num) continue;
      sampled.push_back(orig[i]);
    }
  }
  else
  {
    sampled.resize(orig.size());
    std::copy(orig.begin(), orig.end(), sampled.begin());
  }
}

void SupPar2Vec::run_train(int thread_id)
{
  boost::random_device rd;
  boost::random::mt19937 rng(rd());

  UniformRandom unif_rand(rng, 0.0, 1.0);
  IntRandom int_rand(rng, 0, m_params.L-1);

  const int mb_sz = 10000;      // read this amount of instances from the HDF5 file each time

  real lr = m_params.lr;
  const long long M = !m_is_infer?m_hCorpus->get_num_train_instances():m_hCorpus->get_num_test_instances();
  const long long max_sentences = (long long) ceil(M/(real)m_num_threads);
  long long sentences_seen,last_sentences_seen;

  std::vector<long long> relevant_labels;
  std::vector<long long> irrelevant_labels;

  std::vector<long long> random_label_indices(m_params.L);
  std::iota( random_label_indices.begin(), random_label_indices.end(), 0 );

  std::unique_ptr<real[]> hid(new real[m_params.hid_dim]);
  std::unique_ptr<real[]> grad(new real[m_params.hid_dim]);

  real *W0_(nullptr), *D0_=(nullptr);

  D0_ = D0.get(); // document embeddings
  if(m_is_train_labeled_data)
    W0_ = W0.get(); // label embeddings

  sentences_seen = 0;
  int base_sentence_idx = max_sentences*thread_id;
  int last_idx = std::min(M, base_sentence_idx+max_sentences);
  //LOG(INFO) << "Thread " << thread_id << ": " << base_sentence_idx << " -> " << last_idx-1;

  std::vector<int> label_desc_tokens(0);

  std::vector<int> sampled_doc_word_seq(0);
  std::vector<int> sampled_label_desc_tokens(0);

  int iter_start = m_params.sentences_seen_actual/(real)m_params.trM;
  unsigned int paragraph_idx = 0;
  for(int iter = iter_start; iter < m_params.num_iters; ++iter)
  {
    last_sentences_seen = 0; sentences_seen = 0;
    for(int stx_pos=base_sentence_idx; stx_pos < last_idx; stx_pos += mb_sz)
    {
      int num_instances = (stx_pos + mb_sz >= last_idx)?(last_idx-stx_pos):mb_sz;

      std::vector<std::vector<int>> words_in_docs(num_instances, std::vector<int>(0));    // each vector contains a word sequence for a document
      std::vector<std::vector<int>> labels_in_docs(num_instances, std::vector<int>(0));   // a set of labels

      // load a set of instances from the dataset
      m_corpus_set[thread_id]->getTrainInstances(stx_pos, num_instances, words_in_docs, labels_in_docs);

      // shuffle instance indices
      std::vector<unsigned int> inst_indices(num_instances);
      std::iota(inst_indices.begin(), inst_indices.end(), 0);
      std::shuffle(inst_indices.begin(), inst_indices.end(), rng);

      for(const unsigned int& inst_idx : inst_indices)
      {
        memset(hid.get(), 0, m_params.hid_dim*sizeof(real));
        memset(grad.get(), 0, m_params.hid_dim*sizeof(real));

        paragraph_idx = stx_pos+inst_idx;

        std::vector<int>& doc_word_seq = words_in_docs[inst_idx];
        std::vector<int>& labels = labels_in_docs[inst_idx];
        CHECK(doc_word_seq.size() > 0 && labels.size() > 0);

        sample_words(doc_word_seq, sampled_doc_word_seq, m_params.sample, unif_rand);    // sample out words by frequency

        if(m_params.alpha > 0 && m_is_train_labeled_data)
        {
          // WARP Loss; Learn a joint space of paragraph vectors and label vectors
          std::shuffle(labels.begin(), labels.end(), rng);

          const real avg_loss = compute_warp_loss(int_rand, lr*m_params.alpha, m_params.doc_max_norm, m_params.label_max_norm, random_label_indices, paragraph_idx, labels, hid.get(), grad.get(), relevant_labels, irrelevant_labels);
          CHECK_EQ(relevant_labels.size(),irrelevant_labels.size());
          CHECK(avg_loss>=0);
        }

        if(m_params.beta > 0)
        {
          // distributed bag of words, hierarchical softmax and/or negative sampling
          distributed_memory(lr*m_params.beta, m_params.word_max_norm, m_params.doc_max_norm, rng, sampled_doc_word_seq, D0_+paragraph_idx*m_params.d, hid.get(), grad.get(), false);
        }

        // If we have descriptions of labels
        if(m_params.gamma > 0 && m_is_train_labeled_data)
        {
          // update label representations using their textual description
          for(const long long &label_idx : relevant_labels)
          {
            m_hCorpus->getWordSeqFromLabelDesc(label_idx, label_desc_tokens);
            sample_words(label_desc_tokens, sampled_label_desc_tokens, m_params.sample, unif_rand);

            distributed_memory(lr*m_params.gamma, m_params.word_max_norm, m_params.label_max_norm, rng, sampled_label_desc_tokens, W0_+label_idx*m_params.d, hid.get(), grad.get(), false);
          }

          for(const long long &label_idx : irrelevant_labels)
          {
            m_hCorpus->getWordSeqFromLabelDesc(label_idx, label_desc_tokens);
            sample_words(label_desc_tokens, sampled_label_desc_tokens, m_params.sample, unif_rand);

            distributed_memory(lr*m_params.gamma, m_params.word_max_norm, m_params.label_max_norm, rng, sampled_label_desc_tokens, W0_+label_idx*m_params.d, hid.get(), grad.get(), false);
          }
        }

        sentences_seen++;

        // Output training course
        if(sentences_seen - last_sentences_seen > 100)
        {
          num_processed += (sentences_seen - last_sentences_seen);
          m_params.sentences_seen_actual += sentences_seen - last_sentences_seen;
          last_sentences_seen = sentences_seen;
          if (m_verbose)
          {
            std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
            fprintf(stdout,"%cLearningRate: %f Progress: %.2f%% Processing instances/sec: %.2f ", 13, lr, 
              m_params.sentences_seen_actual / (real)M * 100,
              num_processed / (real) std::chrono::duration_cast<std::chrono::seconds>(now-m_start_time).count() );
            fflush(stdout);
          }
        }
      }
    }
  }

  n_finished_++;

  if(n_finished_ == m_num_threads && !m_is_pretrain)
  {
    model_save_timer.cancel();
    valid_check_timer.cancel();
    std::raise(SIGINT); 
  }
}

void SupPar2Vec::createTable()
{
  long long train_words_pow = 0;
  real d1, power = 0.75;
  std::vector<vocab_ptr>& vocabulary = m_hCorpus->getVocabulary(); 

  m_table = new int[table_size];

  for (long long a = 0; a < m_params.V; a++) train_words_pow += pow(vocabulary[a]->m_freq, power);
  int i = 0;
  d1 = pow(vocabulary[i]->m_freq, power) / (real)train_words_pow;

  for (long long a = 0; a < table_size; a++)
  {
    m_table[a] = i;
    if (a / (real)table_size > d1)
    {
      i++;
      d1 += pow(vocabulary[i]->m_freq, power) / (real)train_words_pow;
    }
    if (i >= m_params.V) i = m_params.V - 1;
  }
}

void SupPar2Vec::export_vectors(std::string filepath)
{
  const long long is_train_labeled_data = m_is_train_labeled_data?1:0;

  // Store word vectors and paragraph vectors 
  std::ofstream file(filepath, std::ios::out | std::ios::binary);

  CHECK(file.is_open()) << "Failed to open the file: " << filepath;

  // store an auxiliary variable first
  file.write((char *) &(is_train_labeled_data), sizeof(long long));

  // store the actual parameters
  file.write((char *) &(m_params.d), sizeof(long long));
  file.write((char *) &(m_params.V), sizeof(long long));
  file.write((char *) U0.get(), m_params.d*(m_params.V+2)*sizeof(real));

  file.write((char *) &(m_params.d), sizeof(long long));
  if(m_is_infer)
  {
    file.write((char *) &(m_params.tsM), sizeof(long long));
    file.write((char *) D0.get(), m_params.d*m_params.tsM*sizeof(real));
  }
  else
  {
    file.write((char *) &(m_params.trM), sizeof(long long));
    file.write((char *) D0.get(), m_params.d*m_params.trM*sizeof(real));
  }

  if(m_is_train_labeled_data)
  {
    file.write((char *) &(m_params.d), sizeof(long long));
    file.write((char *) &(m_params.d), sizeof(long long));
    file.write((char *) V.get(), m_params.d*m_params.d*sizeof(real));

    file.write((char *) &(m_params.d), sizeof(long long));
    file.write((char *) &(m_params.L), sizeof(long long));
    file.write((char *) W0.get(), m_params.d*m_params.L*sizeof(real));
  }

  file.close();

  LOG(INFO) << "Model parameters are stored under the following path: " << filepath;
}

const Parameters& SupPar2Vec::getParams() const
{
  return m_params;
}

void SupPar2Vec::setParamLoaded(bool loaded)
{
  m_loaded_params = loaded;
}

void SupPar2Vec::set_max_iterations(int num_iters)
{
  m_num_iters = num_iters;
  m_params.num_iters = num_iters;
}

void SupPar2Vec::set_verbose(int verbose)
{
  m_verbose = verbose;
} 

void SupPar2Vec::set_model_save_path(std::string model_path)
{
  m_model_save_path = model_path;
}

void SupPar2Vec::set_export_interm_model_interval(int intv)
{
  m_export_interm_model_interval = intv;
}

void SupPar2Vec::set_check_validation_err_interval(int intv)
{
  m_check_validation_err_interval = intv;
}
