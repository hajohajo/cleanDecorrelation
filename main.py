import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from IOTools import read_root_to_dataframe, training_variables
from hep_ml.preprocessing import IronTransformer
from sklearn.preprocessing import MinMaxScaler
from hep_ml.uboost import uBoostBDT, uBoostClassifier
from hep_ml.reweight import BinsReweighter
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from neural_networks import get_disco_classifier, get_autoencoder
from plotting import initialize_plotting, classifierVsX, createROCplot, createMVADistribution, create2DMVAdistribution
import numpy as np
import tensorflow as tf
from sklearn.tree import export_text
even = False

def main():
    baseUrl = "/work/hajohajo/TrainingFiles3/"
    frame = read_root_to_dataframe(baseUrl)
    frame.loc[:, "isSignal"] = frame.loc[:, "eventType"] == 0
    frame = frame.loc[frame.TransverseMass<=500.0, :]
    if not even:
        training_frame = frame.loc[(frame.EventID % 2 == 0), :]
        test_frame = frame.loc[(frame.EventID % 2 != 0), :]
    else:
        training_frame = frame.loc[(frame.EventID % 2 != 0), :]
        test_frame = frame.loc[(frame.EventID % 2 == 0), :]

    bkg = training_frame.loc[(training_frame.loc[:, "isSignal"] == 0), :] #.sample(n=100000, replace=True)
    signal = training_frame.loc[(training_frame.loc[:, "isSignal"] == 1), :].sample(n=bkg.shape[0]) #.sample(n=100000)
    training_frame = signal.append(bkg)
    training_frame = training_frame.sample(frac=1.0)

    test_frame = test_frame.sample(frac=1.0)
    training_targets = training_frame.loc[:, "isSignal"]

    minValues = training_frame.loc[:, training_variables].quantile(0.01).to_numpy()
    maxValues = training_frame.loc[:, training_variables].quantile(0.99).to_numpy()

    discoclassifier, preclassifier, classifier = get_disco_classifier(8, minValues, maxValues)
    # weights = 2*np.ones(training_frame.shape[0])
    # weights[training_targets == 1] = 0

    weights = np.zeros(training_frame.shape[0])
    weighter = BinsReweighter(n_bins=500, n_neighs=3)
    weighter.fit(training_frame.loc[training_targets==0, "TransverseMass"], training_frame.loc[training_targets==1, "TransverseMass"])
    weights[training_targets==0] = 2.0*weighter.predict_weights(training_frame.loc[training_targets==0, "TransverseMass"])
    training_weights = np.ones(training_frame.shape[0])
    training_weights[training_targets==0] = weighter.predict_weights(training_frame.loc[training_targets==0, "TransverseMass"])

    # weights = weighter.predict_weights(training_frame.loc[training_targets==0, "TransverseMass"])


    disco_targets = np.column_stack([training_targets, training_frame.loc[:, "TransverseMass"], weights])
    disco_targets = np.column_stack([training_targets, training_frame.loc[:, "TransverseMass"], np.ones(training_frame.shape[0])])

    # transformer = IronTransformer().fit(training_frame.loc[:, training_variables])
    # transformer = MinMaxScaler().fit(training_frame.loc[:, training_variables])
    # transformed_training = transformer.transform(training_frame.loc[:, training_variables])
    # transformed_testing = transformer.transform(test_frame.loc[:, training_variables])
    transformed_training = training_frame.loc[:, training_variables]
    transformed_testing = test_frame.loc[:, training_variables]

    earlyStopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=250, verbose=1, mode='auto',
        baseline=None, restore_best_weights=False
    )

    #reduce learning rate on plateaus
    reduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=20, verbose=1, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=1e-8
    )

    preclassifier.fit(
        # transformed_training, training_targets,
        {'feature': transformed_training, 'label': training_targets},

        epochs=10,
        batch_size=int(8192 / 16),
        sample_weight=training_weights,
        # callbacks=[earlyStopping, reduceLROnPlateau],
        validation_split=0.1
    )
    #
    initialize_plotting()
    # test_frame.loc[:, "prediction"] = classifier.predict(transformed_testing)
    # create2DMVAdistribution(test_frame, name="toyAdversarial_pre")

    # discoclassifier.fit(
    #     # transformed_training, disco_targets,
    #     {'feature':transformed_training, 'label':disco_targets},
    #     epochs=10,
    #     batch_size=int(8192/8),
    #     sample_weight=training_weights,
    #     callbacks=[earlyStopping, reduceLROnPlateau],
    #     validation_split=0.1
    # )

    discoclassifier.fit(
        # transformed_training, disco_targets,
        {'feature': transformed_training, 'label': disco_targets},
        epochs=500,
        batch_size=int(8192/8),
        sample_weight=training_weights,
        # callbacks=[reduceLROnPlateau],
        callbacks=[earlyStopping, reduceLROnPlateau],
        validation_split=0.1
    )

    #
    # base_tree = DecisionTreeClassifier(
    #     max_depth=20,
    #     min_samples_leaf=0.01,
    #     random_state=0
    # )
    #
    # classifier = uBoostBDT(
    #     base_estimator=base_tree,
    #     uniform_features=["TransverseMass"],
    #     uniform_label=0,
    #     learning_rate=0.5,
    #     n_estimators=100,
    #     uniforming_rate=1.0,
    #     target_efficiency=0.5
    # )
    #
    # classifier = uBoostClassifier(
    #     base_estimator=base_tree,
    #     uniform_features=["TransverseMass"],
    #     uniform_label=0,
    #     n_estimators=100,
    #     subsample=0.1,
    #     learning_rate=0.5,
    #     # uniforming_rate=0.1,
    #     n_threads=12,
    #     efficiency_steps=24
    # )


    # classifier.fit(training_frame.loc[:, training_variables], training_targets)
    # test_frame.loc[:, "prediction"] = classifier.predict_proba(test_frame.loc[:, training_variables])[:, 1]
    # classifier.fit(transformed_training, training_targets)
    # test_frame.loc[:, "prediction"] = classifier.predict_proba(transformed_testing)[:, 1]

    # print(len(classifier.estimators_))
    # print(classifier.estimator_weights_)
    # print(classifier.estimators_[0].get_params())
    # print(export_text(classifier.estimators_[0], show_weights=True))

    # test_frame.loc[:, "prediction"] = classifier.predict(test_frame.loc[:, training_variables])
    test_frame.loc[:, "prediction"] = classifier.predict(transformed_testing)

    classifierVsX(test_frame, "TransverseMass", np.linspace(0.0, 1000.0, 51), plotName="toyAdversarial")
    createMVADistribution(test_frame, name="toyAdversarial")
    createROCplot(test_frame, name="toyAdversarial")
    create2DMVAdistribution(test_frame, name="toyAdversarial")


    if even:
        classifier.save("models/evenModel")
    else:
        classifier.save("models/oddModel")

if __name__ == "__main__":
    main()
