```
def sample (model, test_loader, samples = 16, fname1 = None, fname2 = None, 
            batches=1, save_ind_files=False,flag=0,indices_hist=True,
           show_codes=False):
    model.eval()
    e=flag
    indices_list=[]
    batches=min (len (test_loader), batches)
    print ("Number of batches produced: ", batches)
    with torch.no_grad():

        for batch_idx, (x) in enumerate(tqdm(test_loader)):

            x = x.to(DEVICE)
            
            x_hat, indices, commitment_loss,    = model(x)

            with torch.no_grad():
                x_enc =model.encode(x)
             
            with torch.no_grad():
                z, indices=model.encode_z(x)
            
            with torch.no_grad():
                x_hat_snapped, z_quant =  model.decode_snapped(z,  )
    
            samples= min([samples, x.shape[0]])
             
            x_hat=unscale_image(x_hat)
           
            x=unscale_image(x)
            
            indices_list.append (indices.cpu().detach().flatten().numpy())

            if save_ind_files:
               
                for iu in range (samples):
                    fname2= prefix+ f"recon_samples_{e}_{batch_idx}_{iu}.png"
                    print ("Save individual samples ", fname2)

                    image_sample = to_pil(  x[iu,:].cpu()  )

                    image_sample.save(f'{fname2}', format="PNG",  subsampling=0  )
                    
                    fname2= prefix+ f"recon_samples_{e}_{batch_idx}_{iu}.png"
                    print ("Save individual samples ", fname2)

                    image_sample = to_pil( x_hat[iu,:].cpu()  )

                    image_sample.save(f'{fname2}', format="PNG",  subsampling=0  )
                    
                    plt.imshow (image_sample)
                    plt.axis('off')
                    plt.show()
                    if show_codes:
                        
                        plt.plot (indices[iu,:].cpu().detach().flatten().numpy(),label='Codebook vectors')
                        plt.legend()
                        plt.xlabel ('Codebook index')
                        plt.ylabel ('Codebook vector ID')
                        plt.show()
                        
                        indices_list.append (indices[iu,:].cpu().detach().flatten().numpy())
                        
                        plt.figure(figsize=(8, 3))
                        plt.imshow (indices[iu,:].cpu().detach().flatten().unsqueeze (0).numpy(), cmap='plasma',
                                   aspect=6.)
                        ax = plt.gca()
 
                        ax.get_yaxis().set_visible(False) 
                        plt.clim (0, num_codebook_vectors)
                        plt.colorbar()
                        plt.show()   
                        
                        plt.imshow (indices[iu,:].cpu().detach().numpy(), cmap='plasma',)
                        plt.clim (0, num_codebook_vectors)
                        plt.colorbar()
                        plt.show() 

            draw_sample_image(x[:samples], "Ground-truth images", fname1)
            draw_sample_image(x_hat[:samples], "Reconstructed images", fname2)

            if batch_idx>=(batches-1):
                
                if indices_hist:
                    indices_list=np.array (indices_list).flatten()
                    
                    n_bins =num_codebook_vectors 

                    # Creating histogram
                    fig, axs = plt.subplots(1, 1,
                                            figsize =(4, 3),
                                            tight_layout = True)

                    axs.hist(indices_list, bins = n_bins, density=True)
                    plt.xlabel("Codebook indices")
                    plt.ylabel("Frequency")
                   
                    plt.show()
```
                break

    return
```
```
