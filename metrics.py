from bertprocess import create_ubuntu_testexamples, create_inputs_targets

def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for i in range(len(y)):
        if y[i] in y_test[i][:k]:
            num_correct += 1
    return num_correct/num_examples



def evaluate_recallbert(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for i in range(len(y)):
        if y[i] in y_test[i][:k]:
            num_correct += 1
    return num_correct/num_examples


def evaluate(test_df,model,max_len):
    result=[[] for i in range(10)]
    for col in range(10):
      print("Loading Response set: "+str(col+1))
      train_ubuntu_examples = create_ubuntu_testexamples(test_df,col,max_len)
      x_train, y_train = create_inputs_targets(train_ubuntu_examples)
      temp=model.predict(x_train)
      for i in temp:
        result[col].append(i[0])
    ymain=list(zip(*result))
    yactual=result[0]
    ymain=[ sorted(i)[::-1] for i in ymain ]
    for n in [1, 2, 5, 10]:
        print('Recall @ ({}, 10): {:g}'.format(n, evaluate_recallbert(yactual, ymain, n)))