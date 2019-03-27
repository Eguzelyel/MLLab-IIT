from sampler import UncertaintySampler

def label(sampler):
    cont = "T" 
    while (cont == "T"): 
        # grab k uncertain samples
        k_indices = sampler.sample(20) 

        # get the labelling process going
        sampler.process_k_edus(k_indices)

        # write current state to a file
        sampler.save() 

        cont = input("continue? T/F ")

    print("======> END <======")
    

if __name__ == '__main__':
    sampler = UncertaintySampler()
    label(sampler)