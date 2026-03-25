from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time

BoardShim.enable_dev_board_logger()
params = BrainFlowInputParams()
board = BoardShim(BoardIds.MUSE_2_BOARD, params)
board.prepare_session()
board.start_stream()
time.sleep(5)
data = board.get_board_data()
board.stop_stream()
board.release_session()

eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD)
print(f"Got {data.shape[1]} samples across {data.shape[0]} channels")
print(f"EEG channel indices: {eeg_channels}")
for ch in eeg_channels:
    print(f"  Channel {ch}: {data[ch][:5]}")
