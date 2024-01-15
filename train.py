import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from model import RIR_model, RIR_model_del
from optimizer_utils.bandRIR import rir_bands
from optimizer_utils.enveloper import envelope_generator, rir_smoothing
from optimizer_utils.helpers import calculate_damping_coeff, calculate_del_K, calculate_K_from_delK


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def mag2db(xcv):
    return 20 * xcv.log10_()

def optimize_stochasticRIR(args):
    # hyper params
    iter_ = 1000
    lr = 0.000003
    env_filter_len = 4095
    signal_gain = None    # dB
    dB_clip = -150
    normalize = True
    # optimization params
    device = 'cuda'
    logging_frequency = 100     # epochs  
    stopping_criterion = 300   # epochs \ must be < 0.5* iter
    accepted_loss = 6       # dB
    good_loss = 2
    converging_loss = 5
    convergence_trials = 30
    model_used = RIR_model_del
    known_data = False  # True for generated Data
    #load data
    data_start = int(args.ds)
    data_count = int(args.de)     # default: None
    data_np = np.load(args.fp, allow_pickle=False)
    rir_data = torch.tensor(data_np[data_start:data_count, :], dtype=torch.float).to(device=device)
    #
    #
    rir_convergence_Counter = 0
    for i in range(rir_data.size(0)):
        print(f"\n---------------- Datapoint number: {i+1} out of {rir_data.size(0)} ----------------")
        
        if known_data:
            # for generated data
            lengths, betas, rir_ = rir_data[i, :3], rir_data[i, 3:9], rir_data[i, 9:]
            labels = rir_bands(rir_, device=device)
            target_K_values = calculate_damping_coeff(betas, lengths)
            print(f"Target Kvalues: {target_K_values}")
        else:
            # for real world data 
            labels = rir_bands(rir_data[i, :], device=device)

        # define counters and storage arrays
        nBands = labels.size(1)
        k_array = torch.zeros((nBands, 3))#.to(device=device)    # store K values for all bands
        band_convergence_counter = 0
        band_giveUp_counter = 0
        #
        for j in range(nBands):
            # for each frequency band
            print(f"\n-------- Frequency Band: {j+1} --------")
            # create label envelope
            l_env = envelope_generator(rir_smoothing(labels[:, j], filter_len=env_filter_len, device=device ), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device)
            # define flags and counters
            not_converged = True
            convergence_flag = False
            giveUp_flag = False
            converge_counter = 0
            best_param_dict = {}
            global_loss = torch.inf
            while not_converged:
                bbest_param_dict = {}
                converge_counter += 1
                print(f"\n---- Trial number: {converge_counter} ---- ")
                # Initialization
                mod = model_used(device=device).to(device=device)
                crit = torch.nn.L1Loss().to(device=device)
                optim = torch.optim.SGD(mod.parameters(),lr=lr)
                init_param_dict = {}
                print("\nInitial model Params:")
                for name, param in mod.named_parameters():
                    if param.requires_grad:
                        print( name,': ', param.data)
                        init_param_dict.update({name:param.data.clone()})
                print("\nOptimization process starts:")
                # Gradient descent
                t_l = []
                early_stopping = 0
                min_loss = torch.inf
                for it in range(iter_):
                    optim.zero_grad()
                    y_hat = mod.forward()
                    x_env = envelope_generator(y_hat, gain=signal_gain, clip_=dB_clip, normalise=normalize, device=device)
                    l = crit(x_env, l_env)
                    l.backward()
                    optim.step()
                    log_l = l.detach().cpu()
                    t_l.append(log_l)
                    # early stopping
                    # early stopping for convergent cases
                    if log_l < min_loss:
                        early_stopping = 0
                        min_loss = log_l
                        # save params
                        bbest_param_dict.update({'min_loss': min_loss})
                        for name, param in mod.named_parameters():
                            if param.requires_grad:
                                bbest_param_dict.update({name: param.data.clone()})
                    else:
                        early_stopping += 1       
                    if early_stopping > stopping_criterion: 
                        break
                    if min_loss < good_loss:
                        break
                    if it%logging_frequency == 0 : print(f'Loss in epoch:{it} is : {l.detach()}')
                # converge case
                if min_loss < accepted_loss: 
                    print("converged!")
                    not_converged=False   
                    band_convergence_counter += 1
                    convergence_flag = True

                # tried but not converged  
                if converge_counter >= convergence_trials: 
                    print("Give Up!")
                    not_converged=False
                    band_giveUp_counter += 1
                    giveUp_flag = True   

                #
                if min_loss < global_loss:
                    global_loss = min_loss
                    best_param_dict = bbest_param_dict

            print(f"\nUpdated model for band:")
            final_param_collector = {} #torch.zeros((1)).to(device=device)            
            if giveUp_flag:
                if global_loss < converging_loss:
                    print("converged 2!")
                    band_convergence_counter += 1
                # take the best 
                print("best params:", best_param_dict)
                #final_param_collector = torch.concat((final_param_collector, best_param_dict['del_Kx'].view(-1), best_param_dict['del_Ky'].view(-1), best_param_dict['del_Kz'].view(-1)))
                # final_param_collector = torch.concat((final_param_collector, torch.tensor(list(best_param_dict)[1:]).to(device=device) .view(-1) ))
                for key in best_param_dict:
                    if key == 'min_loss':
                        continue
                    else:
                        final_param_collector.update({key: best_param_dict[key]})
            else:
                for name, param in mod.named_parameters():
                    if param.requires_grad:
                        print( name,': ', param.data)
                        final_param_collector.update({name: param.data})
            print(f"\nMin Loss: {best_param_dict['min_loss']} in dB")
            print(f"Final model params for band-{j+1}: {final_param_collector}")
            final_param_tensor = torch.tensor(list(final_param_collector.values()))
            k_array[j, :] = final_param_tensor
            # 
            if model_used == RIR_model_del:
                final_K_values = calculate_K_from_delK(final_param_tensor[0], final_param_tensor[1], final_param_tensor[2])
                print(f"\nFinal K values: Kx: {final_K_values[0]}, Ky: {final_K_values[1]}, Kz: {final_K_values[2]}")                       
        #   
        print(f"\n################################### \n K values for all Bands of RIR-{i+1}: \n {k_array}\n Converged for {band_convergence_counter} bands \n ################################### ")
        if band_convergence_counter == 6 : rir_convergence_Counter += 1 
    print(f"\nTotal Rir converged: {rir_convergence_Counter} out of {rir_data.size(0)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of SPICE model for F0 estimation')
    parser.add_argument('--fp', '-filepath', type=str, default="./real_room_ir.npy", help='file path of data')
    parser.add_argument('--ds', '-dataStart', type=str, default=0, help='start index of data')
    parser.add_argument('--de', '-dataEnd', type=str, default=6, help='end index of data')
    args = parser.parse_args()

    optimize_stochasticRIR(args)
