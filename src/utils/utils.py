import pathlib
import configparser

#dask.config.set({'temporary-directory': '/scratch'})
def get_cluster(log_dir=None, queue_name = 'pi3', cores = 2, memory = '32G', walltime='1:00:00', **kwargs):
    """ Make dask cluster w/ workers = 2 cores, 32 G mem, and 1 hr wall time.

        return cluster, client
    """

    if log_dir is None:
        log_dir = join(getcwd(),'dask_logs')
        makedirs(log_dir, exist_ok=True)

    cluster = SLURMCluster(
                queue = queue_name,
                cores = 2,
                memory = '32G',
                walltime='1:00:00',
                log_directory=log_dir,
                extra=["--lifetime", "55m", "--lifetime-stagger", "4m"])
    client = Client(cluster)

    print(cluster.dashboard_link)

    return cluster, client
