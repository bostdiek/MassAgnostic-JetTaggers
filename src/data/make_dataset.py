# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
from subprocess import call
from process_data import process_data
from get_weights_1d import set_weights
from process_data import DoPCA

project_dir = Path(__file__).resolve().parents[2]


@click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def download_raw_data():
    logger = logging.getLogger(__name__)

    googled = 'https://drive.google.com/file/d/'
    bkg = googled + '1iYu3vTcNcF1Gg5NplW4WmUkM0I2Jto80/view?usp=sharing'
    p2 = googled + '1ShExpnRkDBkKn6SVxTYTCtc9HNjydmfL/view?usp=sharing'
    p3 = googled + '11dWNfgtRWJqbg0uRdyOM1yvHByh1Q3gf/view?usp=sharing'
    p4 = googled + '1mr1_0OmLfCRI-eude6gWI9RjObvFX7jZ/view?usp=sharing'
    remotes = [bkg, p2, p3, p4]

    basedr = 'data/raw/'
    filenames = ['data_bkg.txt',
                 'data_sig_2p.txt',
                 'data_sig_3p.txt',
                 'data_sig_4p.txt'
                 ]
    for loc_file, remote_file in zip(filenames, remotes):
        raw_name = project_dir.joinpath(basedr + loc_file)
        raw_str = raw_name.resolve()
        if raw_name.exists():
            logger.info('The file {0} has already been downloaded'.format(raw_str))
        else:
            logger.info('Downloading {0}'.format(raw_str))
            call('curl -o {0} {1}'.format(raw_str, remote_file),
                 shell=True)
            logger.info('Download finished')


def SplitAndScale():
    basedr = 'data/interim/'
    logger = logging.getLogger(__name__)
    for prong in range(2, 5):
        train_scaled = 'train_scaled_X_{0}p.npy'.format(prong)
        p_name = project_dir.joinpath(basedr + train_scaled)
        name_str = p_name.resolve()

        if p_name.exists():
            logger.info('The file {0} has already been processed'.format(name_str))
        else:
            logger.info('starting processing {0} prong data'.format(prong))
            process_data(prong)
            logger.info('finished processing {0} prong data'.format(prong))


def GetPlaningWeights():
    basedr = 'data/interim/'
    logger = logging.getLogger(__name__)
    for prong in range(2, 5):
        train_scaled = 'train_planing_weights_{0}p.npy'.format(prong)
        p_name = project_dir.joinpath(basedr + train_scaled)
        name_str = p_name.resolve()

        if p_name.exists():
            logger.info('The weigths {0} have already been processed'.format(name_str))
        else:
            logger.info('computing weights {0} prong data'.format(prong))
            set_weights(prong)
            logger.info('finished processing {0} prong data'.format(prong))


def PCA():
    basedr = 'data/interim/'
    logger = logging.getLogger(__name__)
    for prong in range(2, 5):
        train_scaled = 'train_X_PCA_{0}p.npy'.format(prong)
        p_name = project_dir.joinpath(basedr + train_scaled)
        name_str = p_name.resolve()

        if p_name.exists():
            logger.info('The PCA scaling for {0} have already been done'.format(name_str))
        else:
            logger.info('computing PCA {0} prong data'.format(prong))
            DoPCA(prong)
            logger.info('finished processing {0} prong data'.format(prong))


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    # logger.info('downloading the raw data')
    # download_raw_data()
    logger.info('processing the raw data')
    SplitAndScale()
    logger.info('planing the mass')
    GetPlaningWeights()
    logger.info('Do the PCA rotations')
    PCA()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
