import json
import tqdm as tq

from BERT_CRF import *

def my_initialize():
        with open('test_data.json', encoding='utf-8') as fin:
            test_instances = json.load(fin)

        class_dict = {
            'O': 0,
            'B-Cue': 1,
            'I-Cue': 2,
            'B-Content': 3,
            'I-Content': 4,
            'B-Source': 5,
            'I-Source': 6,
            'X': 11
        }

        # Get the inverted class dictionary
        inv_class_dict = {class_dict[k]: k for k in class_dict}

        # Batch size
        batch_size = 2
        # Hidden size of MLP
        hidden_size = 200
        # Learning rate
        lr = 1e-3
        # Max epochs of system
        epochs = 1
        # Patience of system
        max_patience = 10
        # Window of context
        window = 2

        model = BERT_CRF(b_size=batch_size,
                         n_classes_output=len(class_dict),
                         hidden_size=hidden_size,
                         window=window)

        # Preprocess all test batches
        print('Preprocessing test batches')
        test_batches = [test_instances[i * batch_size: (i + 1) * batch_size] for i in
                        tq.trange((len(test_instances) // batch_size) + 1)]
        if not test_batches[-1]:
            test_batches = test_batches[:-1]

        test_batches = [model.preprocess_batch(batch) for batch in tq.tqdm(test_batches)]

        # Load the best state from the file that it is stored
        state = torch.load('system_best_epoch.pth.tar')

        # Load the parameters into the model
        model.load_state_dict(state['model'])

        my_seed = 1
        random.seed(my_seed)
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        gpu_device = 0
        use_cuda = torch.cuda.is_available()
        if (use_cuda):
            torch.cuda.manual_seed(my_seed)
            print("Using GPU")
        else:
            print("NOT Using GPU")

        return model, class_dict, inv_class_dict, test_batches, use_cuda, 0