#ifndef SUP_PAR2VEC_HPP_
#define SUP_PAR2VEC_HPP_

#include "common.hpp"
#include "labeled_corpus.hpp"
#include "hdf5corpus.hpp"
#include "custom_rand.hpp"

typedef float real;

class Parameters 
{
public:
  long long d;    // dimensionality of word vectors
  long long V;    // the number of words
  long long trM;  // the number of training instances
  long long vaM;  // the number of validation instances
  long long tsM;  // the number of test instances
  long long L;    // the number of labels
  int wn;         // window size
  int hid_dim;    // dimensionality of vectors for the context information
  real lr;        // initial learning rate
  int dm;         // use distributed memory model (PV-DM)
  int dbow;       // use distributed bag or words (PV-DBOW)
  int hs;         // use hierarchical softmax if 1
  real margin;    // the margin in the hinge loss function
  real alpha;     // weight objective functions
  real beta;     // weight objective functions
  real gamma;     // weight objective functions
  real sample;    // sampling probability
  int pretrain_iters;    // number of iterations for pretraining of document and word vectors
  real word_max_norm;  // maximum norm of vector
  real doc_max_norm;  // maximum norm of vector
  real label_max_norm;  // maximum norm of vector
  real lambda;
  int vec_concat;
  int negative;   // the number of negative samples
  int num_iters;  // the number of iterations over train data
  int valid_test;
  long long sentences_seen_actual;

  friend std::ostream& operator<<(std::ostream& out, const Parameters& params)
  {
     return out << "\n==== Parameter settings ====\n"
                << "Dimensionality of word representations: " << params.d << "\n"
                << "Number of words: " << params.V << "\n"
                << "Number of training instances: " << params.trM << "\n"
                << "Number of validation instances: " << params.vaM << "\n"
                << "Number of test instances: " << params.tsM << "\n"
                << "Number of labels: " << params.L << "\n"
                << "Size of window: " << params.wn << "\n"
                << "Size of hidden: " << params.hid_dim << "\n"
                << "Number of epochs: " << params.num_iters << "\n"
                << "Number of pretrain epochs: " << params.pretrain_iters << "\n"
                << "Learning rate: " << params.lr << "\n"
                << "Margin: " << params.margin << "\n"
                << "Weight decay on the linear map: " << params.lambda << "\n"
                << "Importance of learning from the ranking function [0-1]: " << params.alpha << "\n"
                << "Importance of learning from the document representations [0-1]: " << params.beta << "\n"
                << "Importance of learning from the label representations [0-1]: " << params.gamma << "\n"
                << "Sample probability: " << params.sample << "\n"
                << "Max norm of word representations: " << params.word_max_norm << "\n"
                << "Max norm of document representations: " << params.doc_max_norm << "\n"
                << "Max norm of label representations: " << params.label_max_norm << "\n"
                << "Use of hierarchical softmax: " << params.hs << "\n"
                << "Use of negative sampling: " << params.negative << "\n"
                << "Evaluating the intermediate model on the test set: " << params.valid_test << "\n"
                << "============================\n";
  }

  template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
      ar & d;
      ar & V;
      ar & trM;
      ar & vaM;
      ar & tsM;
      ar & L;
      ar & wn;
      ar & hid_dim;
      ar & lr;
      ar & dm;
      ar & dbow;
      ar & hs;
      ar & margin;
      ar & alpha;
      ar & beta;
      ar & gamma;
      ar & sample;
      ar & pretrain_iters;
      ar & word_max_norm;
      ar & doc_max_norm;
      ar & label_max_norm;
      ar & lambda;
      ar & vec_concat;
      ar & negative;
      ar & num_iters;
      ar & valid_test;
      ar & sentences_seen_actual;
    }
  };

  class SupPar2Vec
  {
  public:
    SupPar2Vec(
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
              int verbose);

  ~SupPar2Vec();

  static SupPar2Vec *loadModel(const std::string model_name)
  {
    SupPar2Vec *model = new SupPar2Vec();
    std::ifstream in_stream(model_name);
    boost::archive::binary_iarchive iar(in_stream);
    iar >> *model;
    in_stream.close();
    LOG(INFO) << "A model is loaded from " << model_name;
    //LOG(INFO) << model->getParams();
    model->setParamLoaded(true);
    return model;
  }

  const Parameters& getParams() const;
  void setParamLoaded(bool loaded);
  void setCorpus(HDF5Corpus *corpus);
  void set_verbose(int verbose);
  void set_max_iterations(int num_iters);
  void set_model_save_path(std::string model_path);
  void set_export_interm_model_interval(int intv);
  void set_check_validation_err_interval(int intv);

  void init(); 
  void train(int num_threads);
  void pretrain(int num_threads);
  void export_vectors(std::string filepath);
  void infer(int num_iters, int num_threads, int verbose, std::string output_path);

private:
  SupPar2Vec();
  HDF5Corpus *m_hCorpus;
  Parameters m_params;
  std::vector<std::unique_ptr<HDF5Corpus> > m_corpus_set;
  int* m_table;

  const int table_size = 1e+8;
  real m_margin;      // margin in the hinge loss
  real m_alpha;       // weight between hinge loss (alpha) and log loss (1-alpha)
  real m_beta;       // weight between hinge loss (alpha) and log loss (1-alpha)
  real m_gamma;       // weight between hinge loss (alpha) and log loss (1-alpha)
  real m_sample;
  real m_word_max_norm;
  real m_doc_max_norm;
  real m_label_max_norm;
  real m_lambda;
  int m_num_iters;
  int m_num_threads;
  bool m_loaded_params;
  bool m_is_pretrain;   // is it pretraining phase?
  bool m_is_train_labeled_data;
  bool m_is_infer;
  int m_vec_concat;
  int m_pretrain_iters;
  int m_valid_test;
  int m_verbose;
  std::string m_model_save_path;
  long long num_processed;

  mutable boost::atomic<int> n_finished_;
  int m_export_interm_model_interval;
  int m_check_validation_err_interval;

  boost::asio::io_service io;
  boost::asio::strand strand_;
  boost::asio::signal_set signals;
  boost::asio::signal_set timelimit_signals;
  boost::asio::deadline_timer model_save_timer;
  boost::asio::deadline_timer valid_check_timer;

  void distributed_bow(real lr, real word_max_norm, real entity_max_norm, boost::mt19937 &rng, std::vector<int> &tokens, real *entity_vec, real *grad, const bool is_infer);
  void distributed_memory(real lr, real word_max_norm, real entity_max_norm, boost::mt19937 &rng, std::vector<int> &tokens, real *entity_vec, real *hid, real *grad, const bool is_infer);
  real compute_warp_loss(IntRandom &unif_rand, real lr, real doc_max_norm, real label_max_norm, std::vector<long long>& next_random, long long paragraph_idx, std::vector<int> &label_indices, real *hid, real *grad, std::vector<long long>& relevant_labels, std::vector<long long>& irrelevant_labels);

  std::chrono::time_point<std::chrono::system_clock> m_start_time;

  std::unique_ptr<real[]> V;      // mapping matrix
  std::unique_ptr<real[]> W0;     // label vecotrs
  boost::shared_ptr<real[]> D0;     // paragraph vecotrs
  std::unique_ptr<real[]> U0;     // word vectors
  std::unique_ptr<real[]> U1;     // hierarchical softmax output
  std::unique_ptr<real[]> U2;     // negative sampling output

  void sample_words(std::vector<int>& orig, std::vector<int>& sampled, real sample_prob, UniformRandom &unif_rand);
  void normalize(real *vec, real max_norm);
  void run_train(int thread_id);
  void createTable();

  void save_model();
  void compute_validation_cost();
  real* infer_unseen_documents();
  void run_infer_unseen_documents(int thread_id, real *uD0);
  real* infer_unseen_labels();
  void run_infer_unseen_labels(int thread_id, real *uW0);
  void interrupt_handler(const boost::system::error_code& error,int signal_number);
  void timelimit_handler(const boost::system::error_code& error,int signal_number);
  void check_save_time(const boost::system::error_code& ec);
  void check_validation_time(const boost::system::error_code& ec);

  // Allow serialization to access non-public data members.  
  friend class boost::serialization::access; 

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    real *V_=NULL,*W0_=NULL,*D0_=NULL,*U0_=NULL,*U1_=NULL,*U2_=NULL;
    ar & m_verbose;
    ar & m_model_save_path;
    ar & m_params;     
    ar & m_is_train_labeled_data;
    if(Archive::is_loading::value)
    {
      if(m_is_train_labeled_data)
      {
        V_ = new real[m_params.d * m_params.d];    
        W0_ = new real[m_params.d * m_params.L];    

        V.reset(V_);
        W0.reset(W0_);
      }

      if(m_params.negative > 0)
      { 
        U2_ = new real[m_params.hid_dim * m_params.V];    
        U2.reset(U2_);
      }

      if(m_params.hs)
      {
        U1_ = new real[m_params.hid_dim * m_params.V];    
        U1.reset(U1_);
      }

      D0_ = new real[m_params.d * m_params.trM];    
      U0_ = new real[m_params.d * (m_params.V+2)];    

      D0.reset(D0_);
      U0.reset(U0_);
    }
    else
    {
      if(m_is_train_labeled_data)
      {
        V_ = V.get();
        W0_ = W0.get();
      }
      if(m_params.negative > 0)
      {
        U2_ = U2.get();
      }
      if(m_params.hs)
      {
        U1_ = U1.get();
      }
      D0_ = D0.get();
      U0_ = U0.get();
    }

    if(m_is_train_labeled_data)
    {
    ar & boost::serialization::make_array<real>(V_,m_params.d*m_params.d);
    ar & boost::serialization::make_array<real>(W0_,m_params.d*m_params.L);
    }
    if(m_params.negative > 0)
    {
      ar & boost::serialization::make_array<real>(U2_,m_params.hid_dim*m_params.V);
    }
    ar & boost::serialization::make_array<real>(D0_,m_params.d*m_params.trM);
    ar & boost::serialization::make_array<real>(U0_,m_params.d*(m_params.V+2));
    if(m_params.hs)
    {
      ar & boost::serialization::make_array<real>(U1_,m_params.hid_dim*m_params.V);
    }
  }
};

#endif

