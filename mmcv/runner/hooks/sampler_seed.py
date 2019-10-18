from .hook import Hook


class DistSamplerSeedHook(Hook):

    def before_epoch(self, runner):
        if hasattr(runner, 'data_loader'):
            runner.data_loader.sampler.set_epoch(runner.epoch)
        elif hasattr(runner, 'data_loaders'):
            for data_loader in runner.data_loaders:
                data_loader.sampler.set_epoch(runner.epoch)
