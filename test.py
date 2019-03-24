import csv


def writecsv(output, target, qset):
    """
    :param output: predictions of model
    :param target: expected targets

    write outputs along with predictions to a csv
    """
    with open('outputs.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['output', 'target','set'])

        for i in range(len(output)):
            writer.writerow([output[i], target[i], qset[i]])

    print('Logged predictions')


def test(model, test_data, args, save_output=False):
    """

    :param model: keras model
    :param test_data: tuple of (x, y) array
    :param args:
    :return:
    """
    inputs, target = test_data
    loss = model.evaluate(inputs, target, batch_size=64)
    preds = model.predict(inputs, batch_size=64)

    rounded_out = preds.round().squeeze(1).astype(int)
    crcts = target == rounded_out
    crcts = crcts.mean()

    print('Accuracy: ', crcts)
    print('Test loss: ', loss)

    if save_output:
        writecsv(rounded_out, target, inputs[1])
