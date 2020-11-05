import torch
import torchvision.models as models



###### default profiler #########
def default_profile(model, inputs):
    import torch.autograd.profiler as profiler
    # Following this tutorial:
    # https://pytorch.org/tutorials/recipes/recipes/profiler.html
    # this doesn't really give me much insight into the perf of the network architecture
    # we can group by shapes instead but that is still quite fiddly

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            model(inputs)

    # CPU only
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # CPU and shapes
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

    # CPU and memory
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


# leveraging the torchprof lib
def torchprof(model, inputs):
    import torchprof
    # this leverages torchprof lib from: https://github.com/awwong1/torchprof

    with torchprof.Profile(model, use_cuda=True) as prof:
        model(inputs)

    print(prof.display(show_events=False))


if __name__ == '__main__':

    #model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)

    #default_profile(model, inputs)
    #torchprof(model, inputs)

    model = models.resnet34()
    torchprof(model, inputs)
