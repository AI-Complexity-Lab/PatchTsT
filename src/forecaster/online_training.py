# Prepare data for conformal prediction. Train the base model using availiable data up until now.
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from epiweeks import Week
import yaml
import numpy as np
import random
import argparse
from tqdm import tqdm

from utils import ForecasterTrainer, EarlyStopping, pickle_save, decode_onehot, last_nonzero
from data_utils import get_state_train_data, create_window_seqs, create_fixed_window_seqs, prepare_ds, get_state_test_data_xy
from seq2seq import Seq2seq
from transformer import TransformerEncoderDecoder
from patchTST import PatchTST

def custom_load_model(model, pretrained_state_dict):
    model_state = model.state_dict()
    for name, param in pretrained_state_dict.items():
        if name in model_state:
            if param.size() == model_state[name].size():
                model_state[name].copy_(param)
            else:
                print(f"Size mismatch for {name}: expected {model_state[name].size()}, but got {param.size()}. Weight not loaded.")
        else:
            print(f"{name} not found in current model.")
    model.load_state_dict(model_state, strict=False)
    

def prepare_data(params):
    # predict [pred_week + 1, pred_week + weeks_ahead]
    
    # get params
    smooth = params['smooth']
    fix_window = params['fix_window']
    pred_week = Week.fromstring(params['last_train_week'])
    
    seq_scalers = {
        r: StandardScaler()
        for r in params['regions']
    }  # One scaler per state
    ys_scalers = {
        r: StandardScaler()
        for r in params['regions']
    }  # One scaler per state

    region_idx = {r: i for i, r in enumerate(params['regions'])}

    def one_hot(idx, dim=len(region_idx)):
        ans = np.zeros(dim, dtype="float32")
        ans[idx] = 1.0
        return ans.reshape(1, -1)
    
    test_y_avail = True
    if params['test_week'] == params['last_train_week']:
        test_y_avail = False

    seq_length = 0
    
    # define lists
    train_regions, train_metas, train_xs, train_xs_masks, train_ys, train_ys_mask = [], [], [], [], [], []
    test_regions, test_metas, test_xs, test_xs_masks, test_ys, test_ys_mask = [], [], [], [], [], []

    for region in params['regions']:
        #get the DataFrame x(data) and y(label)
        x, y = get_state_train_data(params, region, smooth)
        x_tmp = seq_scalers[region].fit_transform(x.values[:, :-1])
        x = np.concatenate((x_tmp, x.values[:, -1:]), axis=-1)
        y = ys_scalers[region].fit_transform(y)
        
        #x is a array with dimension [batch size, sequence length, number of features]
        #x_mask: [batch size, sequence length, 1]
        #y is a array with dimension [batch size, week_ahead]
        #y_mask: [batch size, week_ahead]
        x, x_mask, y, y_mask = create_fixed_window_seqs(
            x, y, params['data_params']['min_sequence_length'],
            params['weeks_ahead'], params['data_params']['pad_value']) if fix_window else create_window_seqs(
                x, y, params['data_params']['min_sequence_length'],
                params['weeks_ahead'], params['data_params']['pad_value'])

        #train_xs is a list of x [batch size, sequence length, number of features]
        #train_metas is a list of metas [batch size, number of region]
        #train_xs_mask is a list of x_mask [batch size, sequence length, 1]
        #train_ys is a list of y [batch size, week_ahead]
        #train_ys_mask is a list of y_mask [batch size, week_ahead]
        train_regions.extend([region] * x.shape[0])
        train_metas.append(
            np.repeat(one_hot(region_idx[region]), x.shape[0], axis=0))
        train_xs.append(x.astype(np.float32))
        train_xs_masks.append(x_mask.astype(np.float32))
        train_ys.append(y)
        train_ys_mask.append(y_mask)
        
        seq_length = len(x_mask[0])

        if not test_y_avail:
            test_regions.append(region)
            test_metas.append(one_hot(region_idx[region]))
            test_xs.append(x[[-1]].astype(np.float32))
            test_xs_masks.append(x_mask[[-1]].astype(np.float32))
        else:
            x_test, y_test = get_state_test_data_xy(params, region, pred_week, seq_length, smooth)
            x_test = x_test.to_numpy()
            x_tmp = seq_scalers[region].transform(x_test[:, :-1])
            x_test = np.concatenate((x_tmp, x_test[:, -1:]), axis=-1)
            y_test = ys_scalers[region].transform(y_test)
            test_regions.append(region)
            test_metas.append(one_hot(region_idx[region]))
            test_xs.append([x_test.astype(np.float32)])
            test_xs_masks.append([np.zeros(seq_length).astype(np.float32)])
            test_ys.append([y_test])
            test_ys_mask.append([np.ones_like(y_test)])

    # construct dataset
    dataset = prepare_ds(train_xs, train_xs_masks, train_ys, train_ys_mask, train_regions, train_metas)
    test_dataset = prepare_ds(test_xs, test_xs_masks, test_ys, test_ys_mask, test_regions, test_metas, test=not test_y_avail)

    # split train dataset into train and validation
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    # Create DataLoader for training data
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params['training_parameters']['batch_size'],
        shuffle=True)

    # Create DataLoader for validation data
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params['training_parameters']['batch_size'],
        shuffle=False)

    # Create DataLoader for test data
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=params['training_parameters']['batch_size'],
        shuffle=False)

    train_xs = np.concatenate(train_xs, axis=0)
    return train_dataloader, val_dataloader, test_dataloader, train_xs.shape[2], ys_scalers, seq_length


def prepare_region_fine_tuning_data(params):
    all_dataloaders = {}
    # predict [pred_week + 1, pred_week + weeks_ahead]
    
    # get params
    smooth = params['smooth']
    fix_window = params['fix_window']
    pred_week = Week.fromstring(params['last_train_week'])
    
    seq_scalers = {
        r: StandardScaler()
        for r in params['regions']
    }  # One scaler per state
    ys_scalers = {
        r: StandardScaler()
        for r in params['regions']
    }  # One scaler per state

    region_idx = {r: i for i, r in enumerate(params['regions'])}

    def one_hot(idx, dim=len(region_idx)):
        ans = np.zeros(dim, dtype="float32")
        ans[idx] = 1.0
        return ans.reshape(1, -1)
    
    test_y_avail = True
    if params['test_week'] == params['last_train_week']:
        test_y_avail = False
    
    for region in params['regions']:
        # define lists
        train_regions, train_metas, train_xs, train_xs_masks, train_ys, train_ys_mask = [], [], [], [], [], []
        test_regions, test_metas, test_xs, test_xs_masks, test_ys, test_ys_mask = [], [], [], [], [], []
        
        x, y = get_state_train_data(params, region, smooth)
        x_tmp = seq_scalers[region].fit_transform(x.values[:, :-1])
        x = np.concatenate((x_tmp, x.values[:, -1:]), axis=-1)
        y = ys_scalers[region].fit_transform(y)

        x, x_mask, y, y_mask = create_fixed_window_seqs(
            x, y, params['data_params']['min_sequence_length'],
            params['weeks_ahead'], params['data_params']['pad_value']) if fix_window else create_window_seqs(
                x, y, params['data_params']['min_sequence_length'],
                params['weeks_ahead'], params['data_params']['pad_value'])

        train_regions.extend([region] * x.shape[0])
        train_metas.append(
            np.repeat(one_hot(region_idx[region]), x.shape[0], axis=0))
        train_xs.append(x.astype(np.float32))
        train_xs_masks.append(x_mask.astype(np.float32))
        train_ys.append(y)
        train_ys_mask.append(y_mask)
        
        seq_length = len(x_mask[0])

        if not test_y_avail:
            test_regions.append(region)
            test_metas.append(one_hot(region_idx[region]))
            test_xs.append(x[[-1]].astype(np.float32))
            test_xs_masks.append(x_mask[[-1]].astype(np.float32))
        else:
            x_test, y_test = get_state_test_data_xy(params, region, pred_week, seq_length, smooth)
            x_test = x_test.to_numpy()
            x_tmp = seq_scalers[region].transform(x_test[:, :-1])
            x_test = np.concatenate((x_tmp, x_test[:, -1:]), axis=-1)
            y_test = ys_scalers[region].transform(y_test)
            test_regions.append(region)
            test_metas.append(one_hot(region_idx[region]))
            test_xs.append([x_test.astype(np.float32)])
            test_xs_masks.append([np.zeros(seq_length, dtype=float)])
            test_ys.append([y_test])
            test_ys_mask.append([np.ones_like(y_test)])

        # construct dataset
        dataset = prepare_ds(train_xs, train_xs_masks, train_ys, train_ys_mask, train_regions, train_metas)
        test_dataset = prepare_ds(test_xs, test_xs_masks, test_ys, test_ys_mask, test_regions, test_metas, test=not test_y_avail)

        # split train dataset into train and validation
        train_size = int(len(dataset) * 0.8)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size])

        # Create DataLoader for training data
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=params['rft_batch_size'],
            shuffle=True)

        # Create DataLoader for validation data
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=params['rft_batch_size'],
            shuffle=False)

        # Create DataLoader for test data
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False)

        train_xs = np.concatenate(train_xs, axis=0)
        all_dataloaders[region] = (train_dataloader, val_dataloader, test_dataloader, train_xs.shape[2], ys_scalers)

    return all_dataloaders


def region_fine_tuning(params, model_state_dict, target_region, all_dataloaders, seq_length):
    device = torch.device(params['device'])
    train_dataloader, val_dataloader, _, x_dim, _ = all_dataloaders[target_region]
    metas_dim = len(params['regions'])
    
    # load pretrained model states
    model = None
    if params['model_name'] == 'seq2seq':
        model = Seq2seq(
            metas_train_dim=metas_dim,
            x_train_dim=x_dim-1,
            device=device,
            weeks_ahead=params['weeks_ahead'],
            hidden_dim=params['model_parameters']['hidden_dim'],
            out_layer_dim=params['model_parameters']['out_layer_dim'],
            out_dim=1
        )
    if params['model_name'] == 'transformer':
        model = TransformerEncoderDecoder(
            input_dim=x_dim-1,
            output_dim=params['weeks_ahead'],
            hidden_dim=params['model_parameters']['hidden_dim'],
            seq_length=seq_length,
            num_layers=params['model_parameters']['num_layers'],
            num_heads=params['model_parameters']['num_heads'],
            num_regions=metas_dim,
            rnn_hidden_dim=params['model_parameters']['rnn_hidden_dim'],
            rnn_layers=params['model_parameters']['rnn_layers'],
        )
    if params['model_name'] == 'Patch_tst':
        model = PatchTST(out_layer_dim=params['model_parameters']['out_layer_dim'],
                hidden_dim=params['model_parameters']['hidden_dim'],
                device = device,
                weeks_ahead=params['weeks_ahead'],
                c_in=x_dim-1,
                target_dim=128,
                patch_len=seq_length,
                stride=seq_length,
                num_patch=1,
                n_layers=3,
                n_heads=16,
                d_model=128,
                shared_embedding=True,
                d_ff=256,                        
                dropout=0.2,
                head_dropout=0.2,
                act='relu',
                head_type='regression',
                res_attention=False
        )
    model = model.to(device)
    for name, param in model.named_parameters():
        if name == 'backbone.W_P.weight':
            print(f"{name}: {param.size()}")

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params['training_parameters']['lr'])

    # create loss function
    loss_fn = nn.MSELoss()

    # create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,
        gamma=params['training_parameters']['gamma'],
        verbose=False)

    # create early stopping
    early_stopping = EarlyStopping(
        patience=50, verbose=False)

    # create trainer
    trainer = ForecasterTrainer(model, params['model_name'], optimizer, loss_fn, device)

    # train model
    for epoch in range(params['rft_epochs']):
        trainer.train(train_dataloader, epoch)
        val_loss = trainer.evaluate(val_dataloader, epoch)
        scheduler.step()
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
    
    model.load_state_dict(early_stopping.model_state_dict)
    return model


def forecast(model, model_name, test_dataloader, device, is_test, true_scale, ys_scalers):
    model.eval()
    predictions = {}
    addition_info = {}
    with torch.no_grad():
        for batch in test_dataloader:
            # get data
            regions, meta, x, x_mask, y, y_mask, weekid = batch
            x_mask = x_mask.type(torch.float)
            regionid = decode_onehot(meta)
            weekid = last_nonzero(weekid)

            if model_name == 'seq2seq':
                # send to device
                meta = meta.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)

                # forward pass
                y_pred, emb = model.forward(x, x_mask, meta, output_emb=True)
                emb = emb.cpu().numpy()
                
            if model_name == 'Patch_tst':
                    # send to device
                meta = meta.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)

                # forward pass
                y_pred = model.forward(x)
                emb = np.zeros(len(regions))
                    
            if model_name == 'transformer':
                # send to device
                regionid = regionid.to(device)
                weekid = weekid.to(device)
                x = x.to(device)
                x_mask = x_mask.to(device)

                # forward pass
                y_pred = model.forward(x, x_mask, regionid, weekid).unsqueeze(-1)
                # empty emb
                emb = np.zeros(len(regions))

            # convert to numpy
            y_pred = y_pred.cpu().numpy()
            
            if is_test:
                y = np.zeros((len(regions), len(y_pred[0])))
            else:
                y_cpu = y.cpu()
                y = y_cpu.numpy()
                y = y[:, :, 0]
            meta = meta.cpu().numpy()

            # use scaler to inverse transform
            for i, region in enumerate(regions):
                if true_scale:
                    predictions[region] = ys_scalers[region].inverse_transform(y_pred[i]).reshape(-1)
                    y_in_true_scale = ys_scalers[region].inverse_transform(y[i].reshape(-1, 1)).reshape(-1)
                    addition_info[region] = (emb[i], y_in_true_scale, meta[i])
                else:
                    predictions[region] = y_pred[i].reshape(-1)
                    addition_info[region] = (emb[i], y[i], meta[i])
    return predictions, addition_info


def load_pretrained_model(params):
    pass


def train_and_forcast(last_train_week, params, pretrained_model_state, train=True):
    params['last_train_week'] = Week.fromstring(last_train_week).cdcformat()
    device = torch.device(params['device'])
    true_scale = params['true_scale']
    # decide if this is the test week
    is_test = False
    if params['test_week'] == params['last_train_week']:
        is_test = True
    
    train_dataloader, val_dataloader, test_dataloader, x_dim, ys_scalers, seq_length = prepare_data(params)
    # train_dataloader, test_dataloader, val_dataloader, x_dim, ys_scalers, seq_length = prepare_data(params)
    metas_dim = len(params['regions'])

    # create model
    model = None
    if params['model_name'] == 'seq2seq':
        model = Seq2seq(
            metas_train_dim=metas_dim,
            x_train_dim=x_dim-1,
            device=device,
            weeks_ahead=params['weeks_ahead'],
            hidden_dim=params['model_parameters']['hidden_dim'],
            out_layer_dim=params['model_parameters']['out_layer_dim'],
            out_dim=1
        )
    if params['model_name'] == 'transformer':
        model = TransformerEncoderDecoder(
            input_dim=x_dim-1,
            output_dim=params['weeks_ahead'],
            hidden_dim=params['model_parameters']['hidden_dim'],
            seq_length=seq_length,
            num_layers=params['model_parameters']['num_layers'],
            num_heads=params['model_parameters']['num_heads'],
            num_regions=metas_dim,
            rnn_hidden_dim=params['model_parameters']['rnn_hidden_dim'],
            rnn_layers=params['model_parameters']['rnn_layers'],
        )
    if params['model_name'] == 'Patch_tst':
        model = PatchTST(out_layer_dim=params['model_parameters']['out_layer_dim'],
                hidden_dim=params['model_parameters']['hidden_dim'],
                device = device,
                weeks_ahead=params['weeks_ahead'],
                c_in=x_dim-1,
                target_dim=128,
                patch_len=seq_length,
                stride=seq_length,
                num_patch=1,
                n_layers=3,
                n_heads=16,
                d_model=128,
                shared_embedding=True,
                d_ff=256,                        
                dropout=0.2,
                head_dropout=0.2,
                act='relu',
                head_type='regression',
                res_attention=False
        )
    model = model.to(device)
        
        
    epochs = params['training_parameters']['epochs']
    
    if pretrained_model_state is None:
        train = True
    
    if train:
        if params['week_retrain'] == False and pretrained_model_state is not None:
            model.load_state_dict(pretrained_model_state)
            epochs = params['week_retrain_epochs']

        # create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=params['training_parameters']['lr'])

        # create loss function
        loss_fn = nn.MSELoss()

        # create scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=params['training_parameters']['gamma'],
            verbose=False)

        # create early stopping
        early_stopping = EarlyStopping(
            patience=50, verbose=False)

        # create trainer
        trainer = ForecasterTrainer(model, params['model_name'], optimizer, loss_fn, device)

        # train model
        for epoch in range(epochs):
            trainer.train(train_dataloader, epoch)
            val_loss = trainer.evaluate(val_dataloader, epoch)
            scheduler.step()
            early_stopping(val_loss, model)
            pretrained_model_state = early_stopping.model_state_dict
            
            if 'backbone.W_P.weight' in pretrained_model_state:
                weight_shape = pretrained_model_state['backbone.W_P.weight'].size()
                print(f"backbone.W_P.weight dimensions: {weight_shape}")
            else:
                print("The specified key does not exist in the pretrained model state.")

            if early_stopping.early_stop:
                break
        # pretrained_model_state = early_stopping.model_state_dict
        print('model training finished')

    custom_load_model(model, pretrained_model_state)
    
    # fine-tuning for each state
    rft_models = {}
    all_dataloaders = prepare_region_fine_tuning_data(params)
    if params['region_fine_tuning'] == True:
        for region in params['regions']:
            rft_models[region] = region_fine_tuning(params, pretrained_model_state, region, all_dataloaders, seq_length)
    
    # predict test
    # model.load_state_dict(torch.load('./models/checkpoint.pt'))
    predictions = {}
    addition_info = {}
    if params['region_fine_tuning'] == True:
        for region in params['regions']:
            rft_model = rft_models[region]
            _, _, region_test_dataloader, _, _ = all_dataloaders[region]
            region_predictions, region_addition_info = forecast(rft_model, params['model_name'], region_test_dataloader, device, is_test, true_scale, ys_scalers)
            predictions[region] = region_predictions[region]
            addition_info[region] = region_addition_info[region]
    else:
        predictions, addition_info = forecast(model, params['model_name'], test_dataloader, device, is_test, true_scale, ys_scalers)
    return predictions, addition_info, pretrained_model_state


def get_params(input_file='1'):
    with open('../../setup/covid_mortality.yaml', 'r') as stream:
        try:
            task_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    with open('../../setup/seq2seq.yaml', 'r') as stream:
        try:
            model_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    # merge params
    params = {**task_params, **model_params}
    
    with open(f'../../setup/exp_params/{input_file}.yaml', 'r') as stream:
        try:
            ot_params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print('Error in reading parameters file')
            print(exc)
    
    # overwrite using online training params
    for key, value in ot_params.items():
        params[key] = value

    params['data_params']['start_week'] = Week.fromstring(params['data_params']['start_week']).cdcformat()
    params['test_week'] = Week.fromstring(str(params['test_week'])).cdcformat()
    
    print('Paramaters loading success.')
    
    return params


def run_online_training(params):
    if params['multi_seed']:
        results = {}
        for seed in params['seeds']:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            starting_week = str(params['pred_starting_week'])
            test_week = str(params['test_week'])
            total_weeks_number = int(params['total_weeks_number'])
            
            base_pred = []
            test_pred = None
            pretrained_model_state = None

            # get base model predictions
            for i in tqdm(range(total_weeks_number)):
                current_week = (Week.fromstring(starting_week) + i).cdcformat()
                if current_week != test_week:
                    predictions, addition_infos, pretrained_model_state = train_and_forcast(current_week, params, pretrained_model_state)
                    base_pred.append((predictions, addition_infos))
                if current_week == test_week:
                    test_pred, _, _ = train_and_forcast(current_week, params, pretrained_model_state)
                    break

            results[seed] = {
                'params': params,
                'base_pred': base_pred,
                'test_pred': test_pred 
            }
        return results
    else:
        random.seed(params['seed'])
        np.random.seed(params['seed'])
        torch.manual_seed(params['seed'])
        
        starting_week = str(params['pred_starting_week'])
        test_week = str(params['test_week'])
        total_weeks_number = int(params['total_weeks_number'])
        
        base_pred = []
        test_pred = None
        pretrained_model_state = None
        
        retrain_freq = params['retrain_freq']
        train = True

        # get base model predictions
        for i in tqdm(range(total_weeks_number)):
            if i % retrain_freq == 0:
                train = True
            else:
                train = False
            current_week = (Week.fromstring(starting_week) + i).cdcformat()
            if current_week != test_week:
                predictions, addition_infos, pretrained_model_state = train_and_forcast(current_week, params, pretrained_model_state, train=train)
                base_pred.append((predictions, addition_infos))
            if current_week == test_week:
                test_pred, _, _ = train_and_forcast(current_week, params, pretrained_model_state, train=train)
                break
        
        results = {
            'params': params,
            'base_pred': base_pred,
            'test_pred': test_pred 
        }
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help="Input file")
    args = parser.parse_args()
    input_file = args.input    # for example: 1
    params = get_params(input_file)

    data_id = int(params['data_id'])
    results = run_online_training(params)
    pickle_save(f'../../results/base_pred/saved_pred_{data_id}.pickle', results)