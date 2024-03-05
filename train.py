import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import os

from model import RIR_model, RIR_model_del
from optimizer_utils.bandRIR import rir_bands
from optimizer_utils.enveloper import envelope_generator, rir_smoothing
from optimizer_utils.helpers import calculate_K, calculate_del_K, calculate_K_from_del, mag2db, eps
from generate_StochasticRIR_del import generate_stochastic_rir_del
from generate_StochasticRIR import generate_stochastic_rir


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def optimize_stochasticRIR(args):
    # hyper params
    iter_ = 1000
    lr = 0.000006
    env_filter_len = 4095
    signal_gain = None    # dB
    dB_clip = -180
    normalize = False
    model_used = RIR_model
    # params
    device = 'cuda'
    logging_frequency = 100     # epochs  
    stopping_criterion = 300   # epochs \ must be < 0.5* iter
    accepted_loss = 1.5#20 #3    # good enough loss to stop searching     # dB
    good_loss = 0.5 #10 #2    # minimum loss required - loss lower than this is arbitrary
    converging_loss = 5#50    # maximum accepted loss
    convergence_trials = 10
    lossUpdate_thresh = 0.05
    env_st = 0#48000   # 
    fs = 48000
    maxTime = (96000-env_st)/fs
    cuton = torch.round(torch.tensor(0.10 * maxTime * fs)).int()
    randW_eps = 1e-1
    # For real RIR
    dName = 'arni'
    known_data = False
    save_K_array = True
    data_np = np.load(args.fp, allow_pickle=False)
    data_start = int(args.ds)
    data_count = int(args.de) if args.de is not None else data_np.shape[0]     # default: None
    rir_data = torch.tensor(data_np[data_start:data_count, :], dtype=torch.float).to(device=device)
    label_names = ['Kx', 'Ky', 'Kz', 'Vol', 'Noise', 'Convergence', 'Min_loss']
    #
    Final_kArray = torch.zeros((1, len(label_names)))
    rir_convergence_Counter = 0
    rir_notConverged_memory = []
    print(f"data used: {data_start}, {data_count}")
    for i in range(rir_data.size(0)):
        print(f"---------------- Datapoint number: {i+1} out of {rir_data.size(0)} ----------------")
        if known_data:
            # for generated data
            lengths, betas, rir_ = rir_data[i, :3], rir_data[i, 3:9], rir_data[i, 9:][env_st: ]
            labels = rir_bands(rir_.to(device=device), device=device)
            target_K_values, t_vol = torch.tensor(np.sort(np.array(calculate_K(betas, lengths)))), torch.prod(lengths)
            print(f"Target Kvalues: {lengths}, {target_K_values}, {t_vol}")
        else:
            # for real world data 
            labels = rir_bands(rir_data[i, :].to(device=device), device=device)
            #labels = rir_data.T
        # define counters and storage arrays
        #print(labels.size())
        #print("labels", rir_data.size())
        nBands = labels.size(1)
        k_array = torch.zeros((nBands, len(label_names)))#.to(device=device)    # store K values for all bands
        band_convergence_counter = 0
        band_convergence_counter2 = 0
        band_giveUp_counter = 0
        #
        for j in range(nBands):
            # for each frequency band
            print(f"-------- Frequency Band: {j+1} --------")
            # create label envelope
            l_env = rir_smoothing(labels[:, j], clip_ = dB_clip, filter_len=env_filter_len, useDefaults=False, device=device, )
            #l_env = edc_matlab(labels[:, j], cuton, clip_ = dB_clip)
            #l_env = env_makerF(rir_smoothing(labels[:, j], filter_len=env_filter_len , device=device), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device, display_plots=False)
            #l_env = edc_matlab(rir_smoothing(labels[:, j], filter_len=env_filter_len , device=device), cuton)
            #
            if known_data:
                l_smoothee = rir_smoothing(rir_.to(device=device),clip_ = dB_clip, filter_len=env_filter_len, useDefaults=False, device=device)
            #
            not_converged  = True
            convergence_flag = False
            giveUp_flag = False
            converge_counter = 0
            best_param_dict = {}
            global_loss = torch.inf
            param_plot_dict = { }
            while not_converged:
                bbest_param_dict = {}
                converge_counter += 1
                pparam_plot_dict = { }
                print(f"\n---- Trial number: {converge_counter} ---- ")
                # Initialization
                #volume = targetVol, 
                mod = model_used(max_time=maxTime, device=device).to(device=device)
                crit = torch.nn.SmoothL1Loss(reduction='mean', beta=2.0).to(device=device)
                # crit = torch.nn.HuberLoss(reduction='mean', delta=3.0).to(device=device)
                #crit = torch.nn.L1Loss(reduction='mean').to(device=device)
                optim = torch.optim.SGD(mod.parameters(),lr=lr)#, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=30)
                init_param_dict = {}
                print("\nInitial model Params:")
                for name, param in mod.named_parameters():
                    if param.requires_grad:
                        print( name,': ', param.data)
                        init_param_dict.update({name:param.data.clone()})
                        pparam_plot_dict.update({ name : [param.data.clone().cpu().numpy()]})
                # plt.figure(989+i*j, figsize=(5,4))
                # plt.plot(env_makerF(generate_stochastic_rir_del(del_Kx=init_param_dict['del_Kx'], del_Ky=init_param_dict['del_Ky'], del_Kz=init_param_dict['del_Kz'], noise_level=init_param_dict['noise_level'],  device=device), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='initial')
                # plt.title('Initial')
                # plt.show()
                # Gradient descent
                t_l = []
                early_stopping = 0
                min_loss = torch.inf
                print("\nOptimization process starts:")
                for it in range(iter_):
                    optim.zero_grad()
                    y_hat = mod.forward()
                    y_hat_ = rir_bands(y_hat, device=device)[:,j]
                    #x_env = mag2db(y_hat + eps)
                    x_env = rir_smoothing(y_hat_, filter_len=env_filter_len, useDefaults=True, device=device, clip_=dB_clip)
                    #x_env = env_makerF(y_hat, filter_len=env_filter_len, gain=signal_gain, clip_=dB_clip, normalise=normalize, device=device)
                    #x_env = env_makerF(y_hat, gain=signal_gain, clip_=dB_clip, normalise=normalize, device=device)
                    #x_env = edc_matlab(y_hat, cuton, clip_ = dB_clip)
                    #if x_env.requires_grad : x_env.register_hook(lambda x : print("x_env: ", x,x_env.grad_fn,x_env.data, torch.any(torch.isnan(x))))
                    # random weighting
                    #random_W = torch.FloatTensor(96000 - env_st).uniform_(1-randW_eps, 1+randW_eps).to(device=device)
                    #l = crit(x_env*random_W, l_env*random_W)
                    l = crit(x_env, l_env)
                    #l = torch.mean(torch.abs(x_env - l_env)*random_W)
                    #l = torch.sqrt(torch.clamp(torch.mean(torch.square(x_env - l_env)*torch.square(random_W)), min=0.0000001))
                    #if torch.isnan(l): break
                    l.backward()
                    optim.step()
                    #scheduler.step(l)
                    log_l = l.clone().detach().cpu()
                    t_l.append(log_l)
                    # plot k values
                    for name, param in mod.named_parameters():
                        if param.requires_grad:
                            pparam_plot_dict[name].append(param.data.clone().cpu().numpy())
                    # early stopping
                    # early stopping for convergent cases
                    #if log_l < min_loss:
                    if min_loss - log_l > lossUpdate_thresh:
                        early_stopping = 0
                        min_loss = log_l
                        # save params
                        bbest_param_dict.update({'min_loss': min_loss})
                        bbest_param_dict.update({'min_loss_epoch': torch.tensor(it)})
                        for name, param in mod.named_parameters():
                            if param.requires_grad:
                                bbest_param_dict.update({name: param.data.clone()})
                    else:
                        early_stopping += 1    
                    
                    
                    if it%logging_frequency == 0 : 
                        print(f'Loss in epoch:{it} is : {np.round(np.float64(l.detach()), decimals=4)}')
                        #print(x_env.size(), l_env.size())
                        if False:
                            plt.figure((it+1)*547, figsize=(10,4))
                            plt.subplot(1,2,1)
                            plt.plot(y_hat[:25000].clone().detach().cpu(), label='model')
                            plt.plot(labels[:25000, j].cpu(), label="target", alpha=0.8)
                            #plt.yscale("log")
                            plt.legend()
                            plt.subplot(1,2,2)
                            plt.plot(x_env.clone().detach().cpu(), label='model')
                            plt.plot(l_env.clone().detach().cpu(), label='target')
                            plt.legend()
                            plt.show()
                    
                    if early_stopping > stopping_criterion: 
                        print('Break: Early Stopping')
                        break
                    if min_loss < good_loss:
                        print('Break: Good enough loss')
                        break
                # converge case
                if min_loss < accepted_loss: 
                    print("Converged!: Accepted Loss")
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
                    param_plot_dict = pparam_plot_dict
                
                print("\nUpdated Param after Trial:")
                for name, param in mod.named_parameters():
                    if param.requires_grad:
                        print( name,': ', param.data)
                

            #print(f"\nFinal Updated model Params:")
            final_param_collector = {} #torch.zeros((1)).to(device=device)
            #print("best", best_param_dict)
            
            if giveUp_flag:
                if global_loss < converging_loss:
                    print("\nConverged_2: by converging loss !")
                    band_convergence_counter += 1
                    band_convergence_counter2 += 1
                    convergence_flag = True
                # take the best 
                print("best params:", best_param_dict)
                for key in best_param_dict:
                        if key == 'min_loss' or key=='min_loss_epoch':
                            continue
                        else:
                            final_param_collector.update({key: best_param_dict[key].detach().cpu()})
                #final_param_collector = torch.concat((final_param_collector, best_param_dict['del_Kx'].view(-1), best_param_dict['del_Ky'].view(-1), best_param_dict['del_Kz'].view(-1)))
            else:
                for name, param in mod.named_parameters():
                    if param.requires_grad:
                        #print( name,': ', param.data)
                        #final_param_collector = torch.concat((final_param_collector, param.data.view(-1)))
                        final_param_collector.update({name: param.data.detach().cpu()})
            
            
            ####################-----PLOTTING-----###################### 
            print(f"Final model params for band-{j+1}: {final_param_collector}")
            final_param_tensor = list(final_param_collector.values())
            print("final param:", final_param_tensor)
            print('Best params:', best_param_dict)
            # final_param_tensor.append(convergence_flag)
            # final_param_tensor.append(best_param_dict['min_loss'])
            # k_array[j, :] = torch.tensor(final_param_tensor)
            # final_K_values = calculate_K_from_del(final_param_tensor[0], final_param_tensor[1], final_param_tensor[1])
            # print(f"Final K values: Kx: {final_K_values[0]}, Ky: {final_K_values[1]}, Kz: {final_K_values[2]}")
            ###
            if model_used == RIR_model_del:
                final_K_values = calculate_K_from_del(final_param_tensor[0], final_param_tensor[1], final_param_tensor[2])
                print(f"\nFinal K values: Kx: {np.round(np.float64(final_K_values[0]), 4)}, Ky: {np.round(np.float64(final_K_values[1]), 4)}, Kz: {np.round(np.float64(final_K_values[2]), 4)}")
                final_K_values.append(final_param_tensor[3])
                # plot K value part
                kValue_plot_dict = {'Kx':[],'Ky':[], 'Kz':[], 'V':[]}
                dKeys = list(param_plot_dict.keys())
                print("dkeyss", dKeys)
                for ik in range(len(param_plot_dict[dKeys[0]])):
                    Kx, Ky, Kz = calculate_K_from_del(param_plot_dict[dKeys[0]][ik], param_plot_dict[dKeys[1]][ik], param_plot_dict[dKeys[2]][ik])
                    kValue_plot_dict['Kx'].append(Kx)
                    kValue_plot_dict['Ky'].append(Ky)
                    kValue_plot_dict['Kz'].append(Kz)
                kValue_plot_dict['V'] = param_plot_dict['V']
            else:
                final_K_values = final_param_tensor
                kValue_plot_dict = param_plot_dict
            final_K_values.append(convergence_flag)
            final_K_values.append(best_param_dict['min_loss'])
            print('Final__K:', final_K_values, final_param_tensor)
            k_array[j, :] = torch.tensor(final_K_values)
            ###
            # print(param_dict)
            #print("best: ", best_param_dict)
            param_keys = list(kValue_plot_dict.keys())
            model_param_keys = list(init_param_dict.keys())
            print('keys', model_param_keys, param_keys, len(kValue_plot_dict[param_keys[0]]))
            ### Temp
            if known_data:
                kkk = target_K_values.to(device=device)#torch.tensor([-0.3275, -0.1038, -0.0313])
                vv = t_vol.to(device=device)# torch.tensor(213.56)
                env_mod, stoch_RIR = generate_stochastic_rir(Kx=kkk[0], Ky=kkk[1], Kz=kkk[2], V=vv, max_time=maxTime, device=device, dual_output=True)
                stoch_bands = rir_bands(stoch_RIR, device=device)
            fig = plt.figure(i+j+1, figsize=(15,10))
            ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
            ax2 = plt.subplot2grid((2,6), (0,2), colspan=4)
            ax3 = plt.subplot2grid((2,6), (1,0), colspan=2)
            ax4 = plt.subplot2grid((2,6), (1,2), colspan=2)
            ax5 = plt.subplot2grid((2,6), (1,4), colspan=2)
            fig.suptitle(f"dSet:{dName}, IR:{i+1}, fBand:{j+1}, minLoss:{best_param_dict['min_loss']}dB")
            #plt.subplot(2,3,1)
            ax1.plot(t_l,)# linestyle='', marker='.')
            ax1.set_ylabel('loss value in dB')
            ax1.set_xlabel('Epochs')
            ax1.set_title("Train Loss")
            #plt.subplot(2,3,2)
            ax2.plot(l_env.detach().cpu(), color='r',linestyle='-',label='target') #,label=f'HabetsTarget_fBand:{j+1}')
            if known_data:
                #temp###
                plt.plot(l_smoothee.cpu(),linestyle='-', label ='GT_Habets')
                #####
                plt.plot(rir_smoothing(stoch_RIR.cpu(), filter_len=env_filter_len, useDefaults=True, clip_=dB_clip), label='GT_Stoch',linestyle='--' )
                plt.plot(rir_smoothing(stoch_bands[:,j].cpu(), filter_len=env_filter_len, useDefaults=True, clip_=dB_clip), label=f'StochGen_fBand:{j+1}',linestyle='--',color='k' )
                #plt.plot(mag2db(env_mod).cpu(),linestyle='--', label='Model generated envelope')
            if model_used == RIR_model_del:
                plt.plot(envelope_generator(generate_stochastic_rir_del(del_Kx=init_param_dict[model_param_keys[0]], del_Ky=init_param_dict[model_param_keys[1]], del_Kz=init_param_dict[model_param_keys[2]],  V=init_param_dict[model_param_keys[3]], max_time=maxTime, device=device), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='initial')
                plt.plot(envelope_generator(generate_stochastic_rir_del(del_Kx=best_param_dict[model_param_keys[0]], del_Ky=best_param_dict[model_param_keys[1]], del_Kz=best_param_dict[model_param_keys[2]], V=best_param_dict[model_param_keys[3]], max_time=maxTime, device=device), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='best')
                # plt.plot(edc_matlab(, cuton))
                # plt.plot(edc_matlab(, cuton))
            else:
                # plt.plot(env_makerF(generate_stochastic_rir(Kx=init_param_dict[model_param_keys[0]], Ky=init_param_dict[model_param_keys[1]], Kz=init_param_dict[model_param_keys[2]],  V=init_param_dict[model_param_keys[3]], max_time=maxTime, device=device), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='initial')
                # plt.plot(env_makerF(generate_stochastic_rir(Kx=best_param_dict[model_param_keys[0]], Ky=best_param_dict[model_param_keys[1]], Kz=best_param_dict[model_param_keys[2]], V=best_param_dict[model_param_keys[3]], max_time=maxTime, device=device), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='best')
                # plt.plot(edc_matlab(generate_stochastic_rir(Kx=init_param_dict[model_param_keys[0]], Ky=init_param_dict[model_param_keys[1]], Kz=init_param_dict[model_param_keys[2]],  V=init_param_dict[model_param_keys[3]], max_time=maxTime, device=device), cuton, clip_ = dB_clip).cpu(), label='initial')
                # plt.plot(edc_matlab(generate_stochastic_rir(Kx=best_param_dict[model_param_keys[0]], Ky=best_param_dict[model_param_keys[1]], Kz=best_param_dict[model_param_keys[2]], V=best_param_dict[model_param_keys[3]], max_time=maxTime, device=device), cuton, clip_ = dB_clip).cpu(), label='best')
                # plt.plot(mag2db(generate_stochastic_rir(Kx=init_param_dict[model_param_keys[0]], Ky=init_param_dict[model_param_keys[1]], Kz=init_param_dict[model_param_keys[2]],  V=init_param_dict[model_param_keys[3]], max_time=maxTime, device=device) + eps).cpu(), label='initial', alpha = 0.8) # linestyle='dotted', alpha = 0.8)
                # plt.plot(mag2db(generate_stochastic_rir(Kx=best_param_dict[model_param_keys[0]], Ky=best_param_dict[model_param_keys[1]], Kz=best_param_dict[model_param_keys[2]], V=best_param_dict[model_param_keys[3]], max_time=maxTime, device=device) + eps).cpu(), label='best',linestyle='--',color='m' )
                # plt.plot(rir_smoothing(generate_stochastic_rir(Kx=init_param_dict[model_param_keys[0]], Ky=init_param_dict[model_param_keys[1]], Kz=init_param_dict[model_param_keys[2]],  V=init_param_dict[model_param_keys[3]], max_time=maxTime, device=device), filter_len=env_filter_len,useDefaults=True, device=device, clip_=dB_clip).cpu(), label='initial', alpha = 0.8) # linestyle='dotted', alpha = 0.8)
                # plt.plot(rir_smoothing(generate_stochastic_rir(Kx=best_param_dict[model_param_keys[0]], Ky=best_param_dict[model_param_keys[1]], Kz=best_param_dict[model_param_keys[2]], V=best_param_dict[model_param_keys[3]], max_time=maxTime, device=device), filter_len=env_filter_len,useDefaults=True, device=device, clip_=dB_clip).cpu(), label='best',linestyle='--',color='m' )
                ax2.plot(rir_smoothing(rir_bands(generate_stochastic_rir(Kx=init_param_dict[model_param_keys[0]], Ky=init_param_dict[model_param_keys[1]], Kz=init_param_dict[model_param_keys[2]],  V=init_param_dict[model_param_keys[3]], noise_level=init_param_dict[model_param_keys[4]], max_time=maxTime, device=device), device=device)[:,j], filter_len=env_filter_len,useDefaults=True, device=device, clip_=dB_clip).cpu(), label='initial',linestyle='-.', alpha = 0.8) # linestyle='dotted', alpha = 0.8)
                ax2.plot(rir_smoothing(rir_bands(generate_stochastic_rir(Kx=best_param_dict[model_param_keys[0]], Ky=best_param_dict[model_param_keys[1]], Kz=best_param_dict[model_param_keys[2]], V=best_param_dict[model_param_keys[3]], noise_level=best_param_dict[model_param_keys[4]], max_time=maxTime, device=device), device=device)[:,j], filter_len=env_filter_len,useDefaults=True, device=device, clip_=dB_clip).cpu(), label='best',linestyle='-.',color='m' )
            #plt.plot(env_makerF(mod.forward().detach(), gain=signal_gain, clip_=dB_clip, normalise=normalize,device=device).detach().cpu(), label='final')
            #plt.plot(edc_matlab(mod.forward().detach(), cuton, clip_ = dB_clip).cpu(), label='final')
            #plt.plot(mag2db(mod.forward().detach() + eps).cpu(), label='final', color='g', alpha=0.8)#,linestyle='-.')
            #plt.plot(rir_smoothing(mod.forward().detach() , filter_len=env_filter_len,useDefaults=True, device=device, clip_=dB_clip).cpu(), label='final', color='g', alpha=0.8)#,linestyle='-.')
            ax2.set_ylabel('Amplitude in dB')
            ax2.set_xlabel('samples')
            ax2.set_title("RIR curves")
            ax2.legend()
            #plt.subplot(2,3,4)
            ax3.plot(kValue_plot_dict[param_keys[0]], label=param_keys[0])
            ax3.plot(kValue_plot_dict[param_keys[1]], label=param_keys[1])
            ax3.plot(kValue_plot_dict[param_keys[2]], label=param_keys[2])
            ax3.scatter(best_param_dict['min_loss_epoch'], best_param_dict[param_keys[0]].cpu(), label=f'best_{param_keys[0]}')
            ax3.scatter(best_param_dict['min_loss_epoch'], best_param_dict[param_keys[1]].cpu(), label=f'best_{param_keys[1]}')
            ax3.scatter(best_param_dict['min_loss_epoch'], best_param_dict[param_keys[2]].cpu(), label=f'best_{param_keys[2]}')
            if known_data:
                plt.hlines(target_K_values[0], xmin=0, xmax=len(t_l), label='target_Kx', color='r', linestyle='--')
                plt.hlines(target_K_values[1], xmin=0, xmax=len(t_l), label='target_Ky', color='k', linestyle='--')
                plt.hlines(target_K_values[2], xmin=0, xmax=len(t_l), label='target_Kz', color='m', linestyle='--')
            #plt.plot(param_plot_dict[param_keys[3]], label=param_keys[3])    # Noise Level
            ax3.set_ylabel('value')
            ax3.set_xlabel('epochs')
            ax3.set_title("k Values")
            ax3.legend(loc='lower left')
            #plt.subplot(2,3,5)
            ax4.plot(kValue_plot_dict[param_keys[3]], label='volume')
            ax4.scatter(best_param_dict['min_loss_epoch'], best_param_dict[param_keys[3]].cpu(), color='m',label='best')
            ax4.set_title('Volume in cubic m')
            if known_data:
                plt.hlines(t_vol, xmin=0, xmax=len(t_l), label='target_volume', color='r', linestyle='--' )
            ax4.set_ylabel('Volume')
            ax4.set_xlabel('epochs')
            ax4.legend()
            #plt.subplot(2,3,6)
            ax5.plot(kValue_plot_dict[param_keys[4]], label='Noise Level')
            ax5.scatter(best_param_dict['min_loss_epoch'], best_param_dict[param_keys[4]].cpu(), color='m',label='best')
            ax5.set_title('NoiseLevel in dB')
            ax5.set_ylabel('dB')
            ax5.set_xlabel('epochs')
            ax5.legend()
            fig.tight_layout()
            rootDir = f"./imgs/{dName}_{str(crit)[:-2]}_{str(data_start)}_{str(data_count)}"
            os.makedirs(rootDir+"/plots", exist_ok=True)
            fig_fName = f"rir_{i+1}_band_{j+1}.jpg"
            plt.savefig(os.path.join(rootDir+"/plots", fig_fName), )
            plt.close()
            #print('fname:',os.path.join(rootDir, fig_fName))
        print(f"################################### \n K values for all Bands: \n {label_names} \n{k_array}\n Converged for {band_convergence_counter} bands \n ################################### ")
        Final_kArray = torch.vstack((Final_kArray, k_array))
        if band_convergence_counter == 6 : rir_convergence_Counter += 1 
        else : rir_notConverged_memory.append((i, band_convergence_counter, band_convergence_counter2))    # save rir not converged index
    print(f"--------------\nTotal Rir converged: {rir_convergence_Counter} out of {rir_data.size(0)} \n Rir not converged: {rir_notConverged_memory}\n--------------")
    if save_K_array : 
        df = pd.DataFrame(Final_kArray[1:, :])
        df.columns = label_names
        f_name = f"{os.path.basename(args.fp).split('.')[0]}_{str(data_start)}_{str(data_count)}_{str(crit)[:-2]}_kValues.txt"
        print('file to save data: ', f_name)
        df.to_csv(os.path.join(rootDir,f_name), index=False)
        #torch.save(Final_kArray, args.fp.split('.')[0] + '_' + data_start+ '_' + data_count + '_K_values.pt')
        #print(f"Max:{df['Min_Loss'].max()} and Min:{df['Min_Loss'].min()}")
        #df.drop(df[df.Min_Loss > 1000].index, inplace=True)
        plt.figure(999999, figsize=(6,4))
        plt.hist(df['Min_loss'],rwidth=0.8) #bins=np.arange(np.round(np.float64(df['Min_Loss'].max()))), 
        #plt.xticks(ticks=np.arange(np.round(np.float64(df['Min_Loss'].max()))),  )
        plt.title(f"Histogram of Minimum loss values, Min:{np.round(np.float64(df['Min_loss'].min()),2)},Max:{np.round(np.float64(df['Min_loss'].max()),2)}")
        plt.tight_layout()
        plt_fName = f"Histogram_{dName}_{str(data_start)}_{str(data_count)}_{str(crit)[:-2]}.jpg"
        plt.savefig(os.path.join(rootDir, plt_fName), )
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of SPICE model for F0 estimation')
    parser.add_argument('--fp', '-filepath', type=str, default="./real_room_ir.npy", help='file path of data')
    parser.add_argument('--ds', '-dataStart', type=str, default=0, help='start index of data')
    parser.add_argument('--de', '-dataEnd', type=str, default=5, help='end index of data')
    args = parser.parse_args()

    optimize_stochasticRIR(args)