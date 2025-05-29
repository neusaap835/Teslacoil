import mido
import argparse
import os.path

# MIDI note number to Arduino pitch string
NOTE_NAMES = {
    21: "NOTE_A0", 22: "NOTE_AS0", 23: "NOTE_B0",
    24: "NOTE_C1", 25: "NOTE_CS1", 26: "NOTE_D1", 27: "NOTE_DS1",
    28: "NOTE_E1", 29: "NOTE_F1", 30: "NOTE_FS1", 31: "NOTE_G1", 32: "NOTE_GS1",
    33: "NOTE_A1", 34: "NOTE_AS1", 35: "NOTE_B1", 36: "NOTE_C2", 37: "NOTE_CS2",
    38: "NOTE_D2", 39: "NOTE_DS2", 40: "NOTE_E2", 41: "NOTE_F2", 42: "NOTE_FS2",
    43: "NOTE_G2", 44: "NOTE_GS2", 45: "NOTE_A2", 46: "NOTE_AS2", 47: "NOTE_B2",
    48: "NOTE_C3", 49: "NOTE_CS3", 50: "NOTE_D3", 51: "NOTE_DS3", 52: "NOTE_E3",
    53: "NOTE_F3", 54: "NOTE_FS3", 55: "NOTE_G3", 56: "NOTE_GS3", 57: "NOTE_A3",
    58: "NOTE_AS3", 59: "NOTE_B3", 60: "NOTE_C4", 61: "NOTE_CS4", 62: "NOTE_D4",
    63: "NOTE_DS4", 64: "NOTE_E4", 65: "NOTE_F4", 66: "NOTE_FS4", 67: "NOTE_G4",
    68: "NOTE_GS4", 69: "NOTE_A4", 70: "NOTE_AS4", 71: "NOTE_B4", 72: "NOTE_C5",
    73: "NOTE_CS5", 74: "NOTE_D5", 75: "NOTE_DS5", 76: "NOTE_E5", 77: "NOTE_F5",
    78: "NOTE_FS5", 79: "NOTE_G5", 80: "NOTE_GS5", 81: "NOTE_A5", 82: "NOTE_AS5",
    83: "NOTE_B5", 84: "NOTE_C6", 85: "NOTE_CS6", 86: "NOTE_D6", 87: "NOTE_DS6",
    88: "NOTE_E6", 89: "NOTE_F6", 90: "NOTE_FS6", 91: "NOTE_G6", 92: "NOTE_GS6",
    93: "NOTE_A6", 94: "NOTE_AS6", 95: "NOTE_B6", 96: "NOTE_C7", 97: "NOTE_CS7",
    98: "NOTE_D7", 99: "NOTE_DS7", 100: "NOTE_E7", 101: "NOTE_F7", 102: "NOTE_FS7",
    103: "NOTE_G7", 104: "NOTE_GS7", 105: "NOTE_A7", 106: "NOTE_AS7", 107: "NOTE_B7",
    108: "NOTE_C8"
}

# Convert milliseconds to musical duration
def ms_to_duration(ms, quarter_note_ms):
    ratio = ms / quarter_note_ms
    if 1.8 <= ratio <= 2.2:
        return 2  # half note
    elif 0.9 <= ratio <= 1.1:
        return 4  # quarter note
    elif 0.45 <= ratio <= 0.55:
        return 8  # eighth note
    elif 0.2 <= ratio <= 0.3:
        return 16  # sixteenth note
    elif ratio >= 3.5:
        return 1  # whole note
    else:
        return 16  # fallback

# Parse MIDI and keep only the highest note at any time point.
# This is a crude approximation to find the melody by removing bass notes
def parse_midi_monophonic_highest(midi_file):
    mid = mido.MidiFile(midi_file)
    tempo = 500000  # default 120 BPM
    ticks_per_beat = mid.ticks_per_beat

    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo = msg.tempo
            break

    events = []
    time = 0
    active_notes = {}

    for track in mid.tracks:
        time = 0
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = time
            elif msg.type in ['note_off', 'note_on'] and msg.note in active_notes:
                start_time = active_notes.pop(msg.note)
                duration_ticks = time - start_time
                duration_ms = mido.tick2second(duration_ticks, ticks_per_beat, tempo) * 1000
                quarter_ms = mido.tick2second(ticks_per_beat, ticks_per_beat, tempo) * 1000
                note_name = NOTE_NAMES.get(msg.note, "REST")
                duration = ms_to_duration(duration_ms, quarter_ms)
                events.append((start_time, msg.note, note_name, duration))

    # Sort events by start time
    events.sort(key=lambda x: x[0])

    # Keep only highest note at each timestamp.
    mono_events = []
    current_time = -1
    group = []

    for event in events:
        start, pitch, note_name, duration = event
        if start != current_time:
            if group:
                highest = max(group, key=lambda x: x[1])  # highest pitch
                mono_events.append((highest[2], highest[3]))  # (note_name, duration)
            group = [(start, pitch, note_name, duration)]
            current_time = start
        else:
            group.append((start, pitch, note_name, duration))

    if group:
        highest = max(group, key=lambda x: x[1])
        mono_events.append((highest[2], highest[3]))

    return mono_events

# Save as melody array
def save_melody_array(events, output_file="melody_output.txt"):
    with open(output_file, "w") as f:
        f.write('#include "notes.h"\n\n')
        f.write("int melody[] = {\n")
        for note, dur in events:
            f.write(f"  {note},{dur},\n")
        f.write("};\n")


# Run it
parser = argparse.ArgumentParser()
parser.add_argument('input', help='MIDI file to process.')
parser.add_argument('-o', help='Output filename.')
args = parser.parse_args()

midi_file = args.input
if args.o:
    output_file = args.o
else:
    # Remove ".mid" or ".midi" extension and replace with .txt
    output_file = args.input.rsplit('.', 1)[0] + '.h'

events = parse_midi_monophonic_highest(midi_file)
save_melody_array(events, output_file)

print(f"\nExported {len(events)} notes")
print(f"Output file: {output_file}")
