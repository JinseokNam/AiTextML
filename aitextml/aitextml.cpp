#include "common.hpp"
#include "hdf5corpus.hpp"
#include "sup_par2vec.hpp"
#include "boost/program_options.hpp" 
#include "boost/filesystem.hpp" 

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);

  bool train(false);
  bool resume(false);
  bool infer(false);

  std::string mode;
  std::string dataset_filepath;

  std::string save_model_filepath;
  std::string load_model_filepath;

  std::string logging_path;

  int dm(0), dbow(0), vec_concat(0);
  int vec_dim(0), window_size(0);
  float learning_rate(0), margin(0), sample(0);
  float alpha(0), beta(0), gamma(0);
  int hs(0), negative(0);
  float word_max_norm(0), doc_max_norm(0), label_max_norm(0), weight_penalty(0);
  int num_iters(0), pretrain(0), valid_test(0), export_model_intv(0), comp_valid_err_intv(0), num_threads(0), verbose(0);

  try
  {
    auto value_range_checker = [](int min, int max, char const * const opt_name){
      return [opt_name, min, max](unsigned short v){ 
        if(v < min || v > max){ 
          throw boost::program_options::validation_error
            (boost::program_options::validation_error::invalid_option_value,
             opt_name, std::to_string(v));
        }
      };
    };

    auto mode_checker = []() {
      return [](const std::string option_input){ 
        std::vector<std::string> mode_list = {"train", "resume", "infer"};
        if(std::find(std::begin(mode_list), std::end(mode_list), option_input) == std::end(mode_list)) {
          throw boost::program_options::validation_error
            (boost::program_options::validation_error::invalid_option_value,
             "mode", option_input);
        }
      };
    };

    boost::program_options::options_description desc("Options");
    desc.add_options()
      ("help","Display help messages")
      ("mode", boost::program_options::value<std::string>(&mode)
                          ->required()
                          ->notifier(mode_checker()), "[train | resume | infer]")
      ("dataset", boost::program_options::value<std::string>(&dataset_filepath)->required(), "Filepath of the corpus in HDF5 format")
      ("pretrain", boost::program_options::value<int>(&pretrain)->default_value(0), "Number of pretraining epochs")
      ("save_train_model", boost::program_options::value<std::string>(&save_model_filepath), "Filepath to save an intermediate model")
      ("load_train_model", boost::program_options::value<std::string>(&load_model_filepath), "Filepath to continue training from an intermediate model")
      ("dim", boost::program_options::value<int>(&vec_dim)->default_value(100), "Dimensionality of vector representations")
      ("window", boost::program_options::value<int>(&window_size)->default_value(5), "Size of window")
      ("learning_rate", boost::program_options::value<float>(&learning_rate)->default_value(0.01,"0.01"), "Learning rate")
      ("dm", boost::program_options::value<int>(&dm)->default_value(0), "Use of distributed memory")
      ("dbow", boost::program_options::value<int>(&dbow)->default_value(1), "Use of distributed bag-of-words")
      ("hs", boost::program_options::value<int>(&hs)->default_value(0), "Use of hierarchical softmax")
      ("negative", boost::program_options::value<int>(&negative)->default_value(1), "Use of negative sampling")
      ("margin", boost::program_options::value<float>(&margin)->default_value(1.0), "Margin in the hinge ranking loss function")
      ("alpha", boost::program_options::value<float>(&alpha)->default_value(1/3.0), "Importance of learning from patterns between instances and labels")
      ("beta", boost::program_options::value<float>(&beta)->default_value(1/3.0), "Importance of learning from documents")
      ("sample", boost::program_options::value<float>(&sample)->default_value(1e-4,"1e-4"), "Sampling probability for frequent words")
      ("word_vec_norm", boost::program_options::value<float>(&word_max_norm)->default_value(0), "Maximum norm of word vectors")
      ("doc_vec_norm", boost::program_options::value<float>(&doc_max_norm)->default_value(0), "Maximum norm of document vectors")
      ("label_vec_norm", boost::program_options::value<float>(&label_max_norm)->default_value(0), "Maximum norm of label vectors")
      ("weight_penalty", boost::program_options::value<float>(&weight_penalty)->default_value(0), "Weight decay for the transformation matrix")
      ("model_export_interval", boost::program_options::value<int>(&export_model_intv)->default_value(0), "Export intermediate model in every X minutes")
      ("comp_valid_err_interval", boost::program_options::value<int>(&comp_valid_err_intv)->default_value(0), "Check model performance on the validation set in every X minutes")
      ("vector_concat", boost::program_options::value<int>(&vec_concat)->default_value(0), "Indicate to concate the context vectors to form the local information (0 if averaging the context vectors)")
      ("valid_test", boost::program_options::value<int>(&valid_test)->default_value(0), "Evaluate the intermediate on the test data (0 if evaluating on the valid data); subset")
      ("num_iters", boost::program_options::value<int>(&num_iters)->default_value(1), "Number of iterations")
      ("verbose", boost::program_options::value<int>(&verbose)->default_value(1), "Verbose (0 to turn off)")
      ("num_threads", boost::program_options::value<int>(&num_threads)
                          ->default_value(1)
                          ->notifier(value_range_checker(1,std::thread::hardware_concurrency(),"num_threads")), "Number of threads in parallel processing")
      ("logging", boost::program_options::value<std::string>(&logging_path), "Filepath to log progress; if not set, log only to standard error");

    boost::program_options::variables_map vm;
    try
    {
      boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

      if(vm.count("help"))
      {
        std::cout << "Basic Command Line Parameter App" << std::endl
                  << desc << std::endl;
        return 0;
      }
      if(vm.count("logging")) 
      {
        google::SetLogDestination(google::INFO, vm["logging"].as<std::string>().c_str());
        FLAGS_alsologtostderr=1;
      }
      else
        FLAGS_logtostderr=1;
      if(vm.count("alpha") || vm.count("beta")) 
      {
        if(vm["alpha"].as<float>() + vm["beta"].as<float>() < 0 || vm["alpha"].as<float>() + vm["beta"].as<float>() > 1)
        {
          throw boost::program_options::error("Sum of alpha and beta should be neither smaller than zero nor greater than one.");
        }
        else
        {
          gamma = 1 - (vm["alpha"].as<float>()+vm["beta"].as<float>());
        }
      }

      boost::program_options::notify(vm);

    }
    catch(boost::program_options::error& e)
    {
      std::cerr << "Option parsing error: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return 1;
    }

    if(mode.compare("train") == 0)  train = 1;
    else if(mode.compare("infer") == 0) infer = 1;
    else if(mode.compare("resume") == 0) resume = 1;

    std::unique_ptr<SupPar2Vec> model;
    std::unique_ptr<HDF5Corpus> corpus(new HDF5Corpus(dataset_filepath));

    if(train || resume)
    {
      if(resume)
      {
        CHECK(!load_model_filepath.empty()) << "The filepath for the model on which we resume training is needed";

        LOG(INFO) << "Load an intermediate model from " << load_model_filepath;
        model.reset(SupPar2Vec::loadModel(load_model_filepath));
        if(model->getParams().sentences_seen_actual/(real)model->getParams().trM < num_iters) model->set_max_iterations(num_iters);
        model->set_export_interm_model_interval(export_model_intv);
        model->set_check_validation_err_interval(comp_valid_err_intv);
      }
      else if(train)
      {
        model.reset(new SupPar2Vec(vec_dim, window_size, learning_rate, dm, dbow, hs, negative, margin, alpha, beta, gamma, sample, word_max_norm, doc_max_norm, label_max_norm, weight_penalty, num_iters, export_model_intv, comp_valid_err_intv, save_model_filepath, vec_concat, pretrain, valid_test, verbose));
      }

      model->setCorpus(corpus.get());
      model->init();

      if(!resume && pretrain>0) model->pretrain(num_threads);
      model->train(num_threads);
    }
    else if(infer)
    {
      CHECK(!load_model_filepath.empty()) << "The filepath for the model on which we resume training is needed";

      LOG(INFO) << "Now loading an intermediate model from " << load_model_filepath;
      model.reset(SupPar2Vec::loadModel(load_model_filepath));
      model->setCorpus(corpus.get());
      model->init();

      LOG(INFO) << "Do inference";

      std::string output_path(load_model_filepath + ".inferred_testset_vectors");
      model->infer(0,                 // number of iterations. If it is 0, then the number of iterations for inference will be same with one for training
                   num_threads,
                   verbose,
                   output_path
                  );
    }

  }
  catch(std::exception& e)
  {
    std::cerr << "Unhandled Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
