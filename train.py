import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os

from model import RIR_model, RIR_model_del
from optimizer_utils.bandRIR import rir_bands
from optimizer_utils.enveloper import envelope_generator, rir_smoothing
from optimizer_utils.helpers import calculate_damping_coeff, calculate_del_K, calculate_K_from_delK
from generate_StochasticRIR_del import generate_stochastic_rir_del


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def mag2db(xcv):
    return 20 * xcv.log10_()

def optimize_stochasticRIR(args):
    # hyper params
    iter_ = 900
    lr = 0.000003
    env_filter_len = 4095
    signal_gain = None    # dB
    dB_clip = -150
    normalize = True
    # optimization params
    device = 'cuda'
    logging_frequency = 100     # epochs  
    stopping_criterion = 400   # epochs \ must be < 0.5* iter
    accepted_loss = 6 #30 #6    # good enough loss to stop searching # dB
    good_loss = 1  # minimum loss required - loss lower than this is arbitrary
    converging_loss = 10 #70 #10    # maximum accepted loss
    convergence_trials = 3
    lossUpdate_thresh = 0.1 #0.5 #0.1
    model_used = RIR_model_del
    #crit_used = torch.nn.L1Loss()
    crit_used = torch.nn.SmoothL1Loss(reduction='mean', beta=2.0)#.to(device=device)
    #crit_used = torch.nn.HuberLoss(reduction='mean', delta=3.0)#.to(device=device)
    known_data = True  # True for generated Data
    save_K_array = True
    env_st = 0    # 
    maxTime = (96000-env_st)/48000
    #load data
    dName = 'ism'
    print(args.fp.split('.')[1])
    data_np = np.load(args.fp, allow_pickle=False)
    data_start = int(args.ds)
    data_count = int(args.de) if args.de is not None else data_np.shape[0]     # default: None
    rir_data = torch.tensor(data_np[data_start:data_count, :], dtype=torch.float).to(device=device)
    #
    #
    label_names = ['Kx', 'Ky', 'Kz', 'Noise', 'Convergence', 'Min_Loss', 'Rir_No']
    Final_kArray = torch.zeros((1, len(label_names)))
    rir_convergence_Counter = 0
    rir_notConverged_memory = []
    print(f"data used: {data_start}, {data_count}")
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
        k_array = torch.zeros((nBands, len(label_names)))#.to(device=device)    # store K values for all bands
        band_convergence_counter = 0
        band_convergence_counter2 = 0
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
                mod = model_used(max_time=maxTime, device=device).to(device=device)
                crit = crit_used.to(device=device)
                optim = torch.optim.SGD(mod.parameters(),lr=lr, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.9, patience=50)
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
                    l = crit(x_env, l_env[env_st:])
                    l.backward()
                    optim.step()
                    scheduler.step(l)
                    log_l = l.detach().cpu()
                    t_l.append(log_l)
                    # early stopping
                    # early stopping for convergent cases
                    #if log_l < min_loss:
                    if min_loss - log_l > lossUpdate_thresh:
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
                    if it%logging_frequency == 0 : print(f'Loss in epoch:{it} ist : {np.round(np.float64(l.detach()), decimals=4)}')
                # converge case
                if min_loss < accepted_loss: 
                    print("converged!")
                    not_converged=False   
                    band_convergence_counter += 1
                    convergence_flag = True  
                elif converge_counter >= convergence_trials: 
                    # tried but not converged
                    print("Give Up!")
                    not_converged=False
                    band_giveUp_counter += 1
                    giveUp_flag = True   

                # check global convergence among trials
                if min_loss < global_loss:
                    global_loss = min_loss
                    best_param_dict = bbest_param_dict

            print(f"\nUpdated model for band:")
            final_param_collector = {} #torch.zeros((1)).to(device=device)            
            if giveUp_flag:
                if global_loss < converging_loss:
                    print("converged 2nd Criteria!")
                    band_convergence_counter += 1
                    band_convergence_counter2 += 1
                    convergence_flag = True
                # take the best 
                print("best params:", best_param_dict)
                for key in best_param_dict:
                    if key == 'min_loss':
                        continue
                    else:
                        final_param_collector.update({key: best_param_dict[key].detach().cpu()})
            else:
                for name, param in mod.named_parameters():
                    if param.requires_grad:
                        print( name,': ', param.data)
                        final_param_collector.update({name: param.data.detach().cpu()})
            print(f"\nMin Loss: {np.round(np.float64(best_param_dict['min_loss']), 2)} in dB")
            print(f"Final model params for band-{j+1}: {final_param_collector}")
            final_param_tensor = list(final_param_collector.values())
            # 
            if model_used == RIR_model_del:
                final_K_values = calculate_K_from_delK(final_param_tensor[0], final_param_tensor[1], final_param_tensor[2])
                print(f"\nFinal K values: Kx: {np.round(np.float64(final_K_values[0]), 4)}, Ky: {np.round(np.float64(final_K_values[1]), 4)}, Kz: {np.round(np.float64(final_K_values[2]), 4)}")
            else:
                final_K_values = final_param_tensor
            final_K_values.append(final_param_tensor[3])
            final_K_values.append(convergence_flag)
            final_K_values.append(best_param_dict['min_loss'])
            final_K_values.append(i+1)
            k_array[j, :] = torch.tensor(final_K_values)

            # plot graph
            rootDir = f"./imgs/{dName}_{str(crit_used)[:-2]}_{str(data_start)}_{str(data_count)}_maxTime_mom_sch"
            plt.figure(2*i*nBands+2*j+1, figsize=(12,4))
            plt.suptitle(f"dSet:{dName}, IR:{i+1}, fBand:{j+1}, minLoss:{np.round(np.float64(best_param_dict['min_loss']), 2)}dB")
            plt.subplot(1,2,1)
            plt.plot(t_l,)# linestyle='', marker='.')
            #plt.ylim([0,100])
            plt.ylabel('loss value in dB')
            plt.xlabel('Epochs')
            plt.title("Train Loss")
            plt.subplot(1,2,2)
            plt.plot(envelope_generator(
                    generate_stochastic_rir_del(
                    del_Kx=init_param_dict['del_Kx'],del_Ky=init_param_dict['del_Ky'],
                    del_Kz=init_param_dict['del_Kz'],noise_level=init_param_dict['noise_level'], max_time=maxTime, device=device),
                    gain=signal_gain, clip_=dB_clip, 
                    normalise=normalize,device=device).detach().cpu(), label='initial')
            plt.plot(l_env.detach().cpu()[env_st:],  label='target')
            plt.plot(envelope_generator(
                generate_stochastic_rir_del(del_Kx=best_param_dict['del_Kx'],
                del_Ky=best_param_dict['del_Ky'], del_Kz=best_param_dict['del_Kz'],
                noise_level=best_param_dict['noise_level'], max_time=maxTime, device=device),
                gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='best prediction')
            plt.plot(envelope_generator(
                mod.forward().detach(), gain=signal_gain,
                clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='final prediction')
            plt.ylabel('Amplitude in dB')
            plt.xlabel('Samples')
            plt.title("RIR envelopes")
            plt.legend()
            fig_fName = f"rir_{i+1}_band_{j+1}.jpg"
            os.makedirs(rootDir+"/results", exist_ok=True)
            plt.savefig(os.path.join(rootDir+"/results", fig_fName), )
            plt.close()
            plt.figure(2*i*nBands+2*j+2, figsize=(6,4))
            plt.title(f"RIR envelopes, dSet:{dName}, IR:{i+1}, fBand:{j+1}, minLoss:{np.round(np.float64(best_param_dict['min_loss']), 2)}dB")
            plt.plot(envelope_generator(generate_stochastic_rir_del(del_Kx=init_param_dict['del_Kx'], del_Ky=init_param_dict['del_Ky'], del_Kz=init_param_dict['del_Kz'],  noise_level=init_param_dict['noise_level'], device=device), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='initial')
            plt.plot(l_env.detach().cpu(),  label='target')
            plt.plot(envelope_generator(generate_stochastic_rir_del(del_Kx=best_param_dict['del_Kx'], del_Ky=best_param_dict['del_Ky'], del_Kz=best_param_dict['del_Kz'],noise_level=best_param_dict['noise_level'],  device=device), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='best prediction')
            plt.plot(envelope_generator(mod.forward().detach(), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='final prediction')
            plt.ylabel('Amplitude in dB')
            plt.xlabel('Samples')
            #plt.title("RIR envelopes")
            plt.legend()
            fig_fName = f"rir_{i+1}_band_{j+1}.jpg"
            os.makedirs(rootDir+"/curves", exist_ok=True)
            plt.savefig(os.path.join(rootDir+"/curves", fig_fName), )
            plt.close()
        # print and store K array
        print(f"\n################################### \n K values for all Bands of RIR-{i+1}: \n {label_names} \n {k_array}\n Converged for {band_convergence_counter} bands \n ################################### ")
        Final_kArray = torch.vstack((Final_kArray, k_array))
        # check band convergence
        if band_convergence_counter == 6 : rir_convergence_Counter += 1
        else : rir_notConverged_memory.append((i+1, band_convergence_counter, band_convergence_counter2))    # save rir not converged index
        
    
    print(f"--------------\nTotal Rir converged: {rir_convergence_Counter} out of {rir_data.size(0)} \n Rir not converged: {rir_notConverged_memory}\n--------------")
    if save_K_array : 
        df = pd.DataFrame(Final_kArray[1:, :])
        df.columns = label_names
        f_name = f"{os.path.basename(args.fp).split('.')[0]}_{str(data_start)}_{str(data_count)}_{str(crit_used)[:-2]}_kValues.txt"
        print('file to save data: ', f_name)
        df.to_csv(os.path.join(rootDir,f_name), index=False)
        #torch.save(Final_kArray, args.fp.split('.')[0] + '_' + data_start+ '_' + data_count + '_K_values.pt')
        #print(f"Max:{df['Min_Loss'].max()} and Min:{df['Min_Loss'].min()}")
        df.drop(df[df.Min_Loss > 1000].index, inplace=True)
        plt.figure(999999, figsize=(6,4))
        plt.hist(df['Min_Loss'],rwidth=0.8) #bins=np.arange(np.round(np.float64(df['Min_Loss'].max()))), 
        #plt.xticks(ticks=np.arange(np.round(np.float64(df['Min_Loss'].max()))),  )
        plt.title(f"Histogram of Minimum loss values, Min:{np.round(np.float64(df['Min_Loss'].min()),2)},Max:{np.round(np.float64(df['Min_Loss'].max()),2)}")
        plt.tight_layout()
        plt_fName = f"Histogram_{dName}_{str(data_start)}_{str(data_count)}_{str(crit_used)[:-2]}.jpg"
        plt.savefig(os.path.join(rootDir, plt_fName), )
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of SPICE model for F0 estimation')
    parser.add_argument('--fp', '-filepath', type=str, default="./rirData/ism_280_multi_low.npy", help='file path of data')
    parser.add_argument('--ds', '-dataStart', type=str, default=0, help='start index of data')
    parser.add_argument('--de', '-dataEnd', type=str, default=None, help='end index of data')
    args = parser.parse_args()

    optimize_stochasticRIR(args)
