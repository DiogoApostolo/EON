Datasets:
	-small_cv_dataset: used to validate the models in the supervised and hybrid approaches
	-small_test_dataset: used to test the models and as the dataset in the lexicon approach, which does not need parameter tuning.

EON ontology:
	-output.owl


Txt Files:
	-cv_lbls2: Contains the labels of each documment in the cv dataset
	-test_lbls2: Contains the labels of each documment in the test dataset


	-cv_pre_processed_documment_sentences2: Contains the documents split by sentences in the cv dataset
	-cv_pre_processed_documments2: Contains the documents in the cv dataset
	-test_pre_processed_documments2: Contains the documents split by sentences in the test dataset
	-test_pre_processed_documment_sentences2: Contains the documents in the test dataset

	-pol_val_cv_documents: Contains the polarity value of OntoSenticNet for each sentence of each document, for the cv dataset
	-pol_val_test_documents: Contains the polarity value of OntoSenticNet for each sentence of each document, for the test dataset
	-swn_negative_cv_documents: Contains the negative value of SentiWordNet for each sentence of each document, for the cv dataset
	-swn_negative_test_documents: Contains the negative value of SentiWordNet for each sentence of each document, for the test dataset
	-swn_positive_cv_documents: Contains the positive value of SentiWordNet for each sentence of each document, for the cv dataset
	-swn_positive_test_documents: Contains the negative value of SentiWordNet for each sentence of each document, for the test dataset

	-sentiment_score_cv_documents: Sentiment score of the lexicon approach using Neg4 function by sentence, for the cv dataset
	-sentiment_score_test_documents: Sentiment score of the lexicon approach using Neg4 function by sentence, for the test dataset



Approches:
	Lexicon:
		-lexicon_approach: Used to run the lexicon approach, can be changed to run with the different aggregation and mapping functions
	Supervised:
		-supervised_svr: Used to run the supervised approach, can be changed to run with Doc2Vec and TF-IDF.
	Hybrid:
		-lexicon_tokens: Used to generate the lexicon features used for both hybrid approaches
		-hybrid_approach: Used to run the hybrid approach which used the polarity values of the lexicon approach using Neg4 and a vectorization using TF-IDF
		-new_hybrid_svr: Used to run the hybrid approach which used the polarity values of Ontosenticnet and Sentiwordnet individually as features and a vectorization using TF-IDF


