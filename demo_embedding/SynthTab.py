from amt_tools.datasets import TranscriptionDataset

import amt_tools.tools as tools

# Regular imports
import numpy as np
import torchaudio
import soundfile as sf
import guitarpro
import librosa
import torch
import jams
import os
import random
import json
import hashlib


def load_audio_file(path):
    """Load audio with torchaudio, falling back to soundfile on TorchCodec/FFmpeg issues."""
    try:
        return torchaudio.load(path)
    except Exception:
        audio_np, sample_rate = sf.read(path, always_2d=True, dtype='float32')
        audio = torch.from_numpy(audio_np.T.copy())
        return audio, sample_rate


# Include the namespace for our tablature note-events
NOTE_TAB_NAMESPACE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gp_to_JAMS', 'note_tab.json')
jams.schema.add_namespace(NOTE_TAB_NAMESPACE)


def load_stacked_notes_jams(jams_path):
    """
    Extract MIDI notes spread across sources (e.g. guitar strings) into a dictionary from a JAMS file.

    Parameters
    ----------
    jams_path : string
      Path to JAMS file to read

    Returns
    ----------
    stacked_notes : dict
      Dictionary containing (slice -> (pitches, intervals)) pairs
    """

    # Load data from the JAMS file
    jam = jams.load(jams_path)

    # Extract all midi note annotations
    note_data_slices = jam.annotations['note_tab']

    # Extract the tempo change annotations
    all_tempos = jam.annotations['tempo'][0].data

    # Initialize a dictionary to hold the notes
    stacked_notes = dict()

    # Loop through the slices of the stack
    for slice_notes in note_data_slices:
        # Initialize lists to hold the pitches and intervals
        pitches, intervals_ticks = list(), list()

        # Loop through the notes pertaining to this slice
        for note in slice_notes:
            # Append the note's fret
            pitches.append(note.value['fret'])
            # Append the note's onset and offset (in ticks)
            intervals_ticks.append([note.time, note.time + note.duration])

        # Convert the pitch and interval lists to arrays
        pitches, intervals_ticks = np.array(pitches), np.array(intervals_ticks)

        # Extract the open string pitch for the string
        string_pitch = slice_notes.sandbox['open_tuning']

        # Add open-string tuning to obtain pitches
        pitches += string_pitch

        # Create an array with same shape as the tick intervals
        intervals = np.zeros(intervals_ticks.shape, dtype=float)

        for tempo_change in all_tempos:
            # Extract relevant tempo information
            tempo = tempo_change.value
            onset = tempo_change.time
            duration = tempo_change.duration

            # Determine the amount of ticks within the tempo boundaries that come before each interval time
            num_ticks = np.maximum(np.minimum(intervals_ticks - onset, duration), 0)

            # Accumulate the time elapsed in seconds before each onset or offset
            intervals += (60 / tempo) * num_ticks / guitarpro.Duration.quarterTime

        # Add the pitch-interval pairs to the stacked notes dictionary under the string entry as key
        stacked_notes.update(tools.notes_to_stacked_notes(pitches, intervals, string_pitch))

    # Re-order keys starting from lowest string and switch to the corresponding note label
    stacked_notes = {librosa.midi_to_note(i) : stacked_notes[i] for i in sorted(stacked_notes.keys())}

    return stacked_notes


class SynthTab(TranscriptionDataset):
    """
    Implements a wrapper for SynthTab (https://synthtab.dev).
    """

    DEV_PARTITIONS = ('acoustic', 'electric_clean', 'electric_distortion', 'electric_distortion_di', 'electric_muted')

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None,
                       profile=None, num_frames=None, audio_norm=np.inf, reset_data=False, store_data=True,
                       save_data=True, save_loc=None, guitars=None, sample_attempts=1, augment_audio=False,
                       include_onsets=False, seed=0, jams_dir=None):
        """
        Initialize an instance of the SynthTab dataset.

        Parameters
        ----------
        See TranscriptionDataset class for others...

        guitars : list of string or None (Optional)
          Names of guitars to include in this instance
        sample_attempts : int (>= 1)
          Number of attempts to sample non-silence
        augment_audio : bool
          Whether to combine and augment the separate microphone signals
        include_onsets : bool
          Whether to include onset activations within the ground-truth
        """

        self.guitars = guitars
        self.sample_attempts = max(1, sample_attempts)
        self.augment_audio = augment_audio
        self.include_onsets = include_onsets
        self.dataset_seed = seed
        self.jams_dir = os.path.abspath(jams_dir) if jams_dir else None

        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile, num_frames,
                         audio_norm, False, reset_data, store_data, save_data, save_loc, seed)

    def get_tracks(self, split):
        """
        Get the tracks associated with a partition of the dataset.

        Parameters
        ----------
        split : string
          Name of the partition from which to fetch tracks

        Returns
        ----------
        tracks : list of strings
          Names of tracks within the given partition
        """

        if self._is_standard_layout():
            return self._get_tracks_standard(split)

        if self._is_dev_layout():
            return self._get_tracks_dev(split)

        raise FileNotFoundError(
            f"Unrecognized SynthTab layout under '{self.base_dir}'. "
            "Expected either train/val directories or at least one dev/full chunk audio "
            "partition with a matching jams directory."
        )

    def get_track_data(self, track, seq_length=None):
        """
        Get the features and ground truth for a track.

        Parameters
        ----------
        track : string
          SynthTab track name
        seq_length : int
          Number of samples to take for the slice

        Returns
        ----------
        data : dict
          Dictionary containing for the track
        """

        # Check if a specific sequence length was given
        if seq_length is None:
            if self.seq_length is not None:
                # Use the global sequence length
                seq_length = self.seq_length

        # Determine the expected path to the track's audio
        audio_path = self.get_feats_dir(track)

        # Empty audio variable to populate
        audio = None

        try:
            # Check if an entry for the audio exists
            if self.save_data and os.path.exists(audio_path):
                # Load and unpack the audio
                audio = torch.load(audio_path)
        except Exception as e:
            # Print offending track to console and regenerate audio
            print(f'Error loading audio for track \'{track}\': {repr(e)}')

        if audio is None:
            # Construct the paths to the track's audio
            audio_paths = self.get_audio_paths(track)

            # Initialize a list to hold all microphone signals
            audio = list()

            for path in audio_paths:
                # Load and normalize the audio
                audio_, fs_ = load_audio_file(path)
                # Extract the first channel
                audio_ = audio_[0].unsqueeze(0)
                # Resample audio to appropriate sampling rate
                audio_ = torchaudio.functional.resample(audio_, fs_, self.sample_rate)

                if self.audio_norm == np.inf or self.audio_norm == torch.inf:
                    # Normalize the audio to the range [-1, 1]
                    audio_ /= audio_.max()
                else:
                    # TODO
                    return NotImplementedError

                # Add the microphone signal to the list
                audio += [audio_]

            # Concatenate microphone signals
            audio = torch.cat(audio)

            if self.save_data:
                # Make sure the top-level pre-processed audio directory exists
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                # Save the pre-processed audio
                torch.save(audio, audio_path)

        # Determine the expected path to the track's ground-truth
        gt_path = self.get_gt_dir(track)

        # Empty ground-truth variable to populate
        stacked_multi_pitch = None

        try:
            # Check if an entry for the ground-truth exists
            if self.save_data and os.path.exists(gt_path):
                # Load and unpack the ground-truth
                ground_truth = tools.load_dict_npz(gt_path)
                # Extract the string-level multi-pitch activations
                stacked_multi_pitch = ground_truth[tools.KEY_MULTIPITCH]

                if self.include_onsets:
                    # Extract the string-level onset activations
                    stacked_onsets = ground_truth[tools.KEY_ONSETS]
        except Exception as e:
            # Print offending track to console and regenerate ground-truth
            print(f'Error loading ground-truth for track \'{track}\': {repr(e)}')

        if stacked_multi_pitch is None:
            # Construct the path to the track's JAMS data
            jams_path = self.get_jams_path(track)

            # Load the notes by string from the JAMS file
            stacked_notes = load_stacked_notes_jams(jams_path)

            # Determine the times associated with each frame of audio
            times = self.data_proc.get_times(audio[0])

            # Represent the string-wise notes as a stacked multi pitch array
            stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

            if self.include_onsets:
                # Obtain onset activations at the string-level for all of the notes
                stacked_onsets = tools.stacked_notes_to_stacked_onsets(stacked_notes, times, self.profile)

            if self.save_data:
                # Make sure the top-level ground-truth directory exists
                os.makedirs(os.path.dirname(gt_path), exist_ok=True)
                # Add the ground-truth to a dictionary that will be saved
                save_data = {tools.KEY_MULTIPITCH : stacked_multi_pitch}

                if self.include_onsets:
                    # Add the onsets to the dictionary
                    save_data.update({tools.KEY_ONSETS : stacked_onsets})

                # Save the pre-computed ground-truth
                tools.save_dict_npz(gt_path, save_data)

        if seq_length is not None:
            # Determine how many audio samples are available
            audio_length = audio[0].shape[-1]

            if audio_length >= seq_length:
                # Initialize a counter for attempts
                attempts_left = self.sample_attempts

                while attempts_left > 0:
                    # Sample a random starting index for the trim
                    sample_start = self.rng.randint(0, audio_length - seq_length + 1)
                    # Determine the frames contained in this slice
                    frame_start = sample_start // self.hop_length
                    frame_end = frame_start + self.num_frames

                    if np.max(stacked_multi_pitch[..., frame_start : frame_end]) > 0:
                        # Non-silence was sampled
                        attempts_left = 0
                    else:
                        # Make another attempt to sample non-silence
                        attempts_left -= 1

                # Trim all microphone signals to the appropriate sequence length
                audio = torch.cat([a[sample_start: sample_start + seq_length].unsqueeze(0) for a in audio])
                # Trim the ground-truth multi-pitch activations to the corresponding frames
                stacked_multi_pitch = stacked_multi_pitch[..., frame_start: frame_end]

                if self.include_onsets:
                    # Trim the ground-truth onset activations to the corresponding frames
                    stacked_onsets = stacked_onsets[..., frame_start: frame_end]
            else:
                # Determine how much padding is required
                pad_total = seq_length - audio_length
                # Pad all microphones with zeros to meet appropriate sequence length
                audio = torch.cat([torch.nn.functional.pad(a, (0, pad_total)).unsqueeze(0) for a in audio])
                # Determine how many frames are missing from the ground-truth
                frames_missing = self.num_frames - stacked_multi_pitch.shape[-1]
                # Pad the ground-truth multi-pitch activations to the corresponding number of frames
                stacked_multi_pitch = np.pad(stacked_multi_pitch, ((0, 0), (0, 0), (0, frames_missing)))

                if self.include_onsets:
                    # Pad the ground-truth onset activations to the corresponding number of frames
                    stacked_onsets = np.pad(stacked_onsets, ((0, 0), (0, 0), (0, frames_missing)))

        audio = audio[self.rng.randint(0, audio.shape[0])].unsqueeze(0)

        # Compute features for the sampled audio snippet
        features = self.data_proc.process_audio(audio.squeeze().numpy())

        # Convert the stacked multi pitch array into tablature
        tablature = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

        # Convert the stacked multi pitch array into a single representation
        multi_pitch = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

        # Add all relevant ground-truth to the dictionary
        data = {tools.KEY_TRACK: track,
                tools.KEY_FS: self.sample_rate,
                tools.KEY_AUDIO: audio,
                tools.KEY_FEATS: features,
                tools.KEY_TABLATURE: tablature,
                tools.KEY_MULTIPITCH: multi_pitch}

        if self.include_onsets:
            # Add the onsets to the ground-truth dictionary
            data.update({tools.KEY_ONSETS : stacked_onsets})

        return data

    def get_audio_paths(self, track):
        """
        Get the paths to the microphone signals of a track.

        Parameters
        ----------
        track : string
          SynthTab track name

        Returns
        ----------
        audio_paths : list of string
          Paths to the various microphone recordings for the track
        """

        track_dir = os.path.join(self.base_dir, track)
        audio_exts = ('.wav', '.flac', '.mp3')
        audio_paths = []

        for root, _dirs, files in os.walk(track_dir):
            for file_name in files:
                if file_name.lower().endswith(audio_exts):
                    audio_paths.append(os.path.join(root, file_name))

        audio_paths.sort()

        return audio_paths

    def get_jams_path(self, track):
        """
        Get the path to the annotations of a track.

        Parameters
        ----------
        track : string
          SynthTab track name

        Returns
        ----------
        jams_path : string
          Path to the JAMS file of the specified track
        """

        if self._is_standard_layout():
            jams_path = os.path.join(self.base_dir, os.path.dirname(track), 'ground_truth.jams')
        else:
            song = os.path.basename(track)
            jams_dir = self._get_jams_song_dir(song)
            if not os.path.isdir(jams_dir):
                raise FileNotFoundError(f"No JAMS directory was found for '{song}' at '{jams_dir}'.")
            jams_candidates = sorted(
                [os.path.join(jams_dir, name) for name in os.listdir(jams_dir) if name.lower().endswith('.jams')]
            )
            if not jams_candidates:
                raise FileNotFoundError(f"No .jams file was found in '{jams_dir}'.")
            jams_path = jams_candidates[0]

        return jams_path

    def get_feats_dir(self, track=None):
        """
        Get the path for the features directory or a track's features.

        Parameters
        ----------
        track : string or None
          Append a track to the directory for the track's features path

        Returns
        ----------
        path : string
          Path to the features directory or a specific track's features
        """

        # Get the path to the directory holding the pre-processed audio
        path = os.path.join(self.save_loc, self.dataset_name(), 'audio')

        if track is not None:
            # Append track name to path if provided
            path = os.path.join(path, f'{track}.npz')

        return path

    @staticmethod
    def available_splits():
        """
        Obtain a list of pre-defined dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset for training/validation stage
        """

        splits = ['train', 'val']

        return splits

    def _is_standard_layout(self):
        return all(os.path.isdir(os.path.join(self.base_dir, split)) for split in self.available_splits())

    def _is_dev_layout(self):
        has_audio_partition = any(os.path.isdir(os.path.join(self.base_dir, name)) for name in self.DEV_PARTITIONS)
        return has_audio_partition and os.path.isdir(self._get_jams_root())

    def _get_jams_root(self):
        return self.jams_dir if self.jams_dir else os.path.join(self.base_dir, 'jams')

    def _get_jams_song_dir(self, song):
        return os.path.join(self._get_jams_root(), song)

    def _get_tracks_standard(self, split):
        tracks = list()
        split_dir = os.path.join(self.base_dir, split)

        for root, dirs, files in os.walk(split_dir):
            if 'ground_truth.jams' in files:
                song = os.path.basename(root)
                tracks += [os.path.join(split, song, guitar) for guitar in dirs
                           if self.guitars is None or guitar in self.guitars]

        return tracks

    def _get_tracks_dev(self, split):
        candidates = self._get_dev_track_candidates()
        cache_path = self._get_dev_track_cache_path(candidates)
        tracks = self._load_dev_track_cache(cache_path)

        if tracks is None:
            tracks = self._validate_dev_track_candidates(candidates)
            self._save_dev_track_cache(cache_path, tracks)

        rng = random.Random(self.dataset_seed)
        rng.shuffle(tracks)

        if not tracks:
            return tracks

        cut = max(1, int(round(len(tracks) * 0.8)))
        if split == 'train':
            return tracks[:cut]
        if split == 'val':
            return tracks[cut:]
        return tracks

    def _get_dev_track_candidates(self):
        candidates = []
        for partition in self.DEV_PARTITIONS:
            partition_dir = os.path.join(self.base_dir, partition)
            if not os.path.isdir(partition_dir):
                continue

            for guitar in sorted(os.listdir(partition_dir)):
                guitar_dir = os.path.join(partition_dir, guitar)
                if not os.path.isdir(guitar_dir):
                    continue
                if self.guitars is not None and guitar not in self.guitars:
                    continue

                for song in sorted(os.listdir(guitar_dir)):
                    song_dir = os.path.join(guitar_dir, song)
                    if not os.path.isdir(song_dir):
                        continue

                    candidates.append((os.path.join(partition, guitar, song), song))

        return candidates

    def _validate_dev_track_candidates(self, candidates):
        tracks = []
        for track, song in candidates:
            jams_dir = self._get_jams_song_dir(song)
            jams_candidates = []
            if os.path.isdir(jams_dir):
                jams_candidates = [
                    os.path.join(jams_dir, name)
                    for name in os.listdir(jams_dir)
                    if name.lower().endswith('.jams')
                ]

            has_jams = bool(jams_candidates)
            matches_profile = False

            if has_jams:
                try:
                    stacked_notes = load_stacked_notes_jams(sorted(jams_candidates)[0])
                    matches_profile = len(stacked_notes) == self.profile.get_num_dofs()
                except Exception:
                    matches_profile = False

            # Some dev/full audio folders do not have a matching JAMS directory.
            # Others have annotations incompatible with the current guitar profile.
            # Filter them out up front so larger train/val runs stay reproducible.
            has_audio = bool(self.get_audio_paths(track))

            if has_audio and has_jams and matches_profile:
                tracks.append(track)

        return tracks

    def _get_dev_track_cache_path(self, candidates):
        if not self.save_loc:
            return None

        payload = {
            'version': 2,
            'base_dir': os.path.abspath(self.base_dir),
            'jams_root': os.path.abspath(self._get_jams_root()),
            'guitars': self.guitars,
            'profile_dofs': self.profile.get_num_dofs() if self.profile is not None else None,
            'seed': self.dataset_seed,
            'tracks': [track for track, _song in candidates],
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]
        return os.path.join(self.save_loc, self.dataset_name(), f'dev-track-index-{digest}.json')

    def _load_dev_track_cache(self, cache_path):
        if cache_path is None or not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            if payload.get('version') != 2:
                return None
            tracks = payload.get('tracks')
            if isinstance(tracks, list) and all(isinstance(track, str) for track in tracks):
                return tracks
        except Exception as e:
            print(f'Error loading SynthTab track index cache \'{cache_path}\': {repr(e)}')

        return None

    def _save_dev_track_cache(self, cache_path, tracks):
        if cache_path is None:
            return

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        payload = {
            'version': 2,
            'base_dir': os.path.abspath(self.base_dir),
            'jams_root': os.path.abspath(self._get_jams_root()),
            'num_tracks': len(tracks),
            'tracks': tracks,
        }
        with open(cache_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def download(save_dir):
        """
        TODO
        """

        return NotImplementedError
