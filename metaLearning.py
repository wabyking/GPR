from sklearn.linear_model import LogisticRegression


selected_models = [
'',
'',
]

def test_model():
    process = Process(params)
    _ = process.get_train()  #waby: in orde to get get the test set properly [build word index parameter], actually.
    val_uncontatenated = process.get_test()
#    train_contatenated = process.get_train(contatenate =1)
    val_contatenated = process.get_test(contatenate =1)
    predicted = emsemble([val_uncontatenated,val_contatenated])    
    
    draw_result(predicted,val_contatenated[1])

if __name__ == '__main__':
	
	# get test data
	process = Process(params)
    _ = process.get_train()  #waby: in orde to get get the test set properly [build word index parameter], actually.
    val_uncontatenated = process.get_test()
#    train_contatenated = process.get_train(contatenate =1)
    val_contatenated = process.get_test(contatenate =1)

    # get all results from the predict models
    results = []
	for model_path in selected_models:
		model = load_model(model_path)
		predicted = model.predict(val[0])
		resutls.append(predicted)

	# logistic regression
	clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(results,val_contatenated[1])
	clf.predict(results)
