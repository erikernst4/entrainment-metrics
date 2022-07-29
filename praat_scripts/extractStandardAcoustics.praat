# Given a sound file, starting and ending points (use 0.0 for both
# values to use the entire file) and the speaker's pitch range,
# extract the standard set of acoustic features:
#
#     + duration in seconds
#     + f0 max
#     + f0 min
#     + f0 mean
#     + f0 median
#     + f0 standard deviation
#     + mean absolute f0 slope
#     + energy max
#     + energy min
#     + energy mean
#     + energy standard deviation
#     + ratio of voiced frames to total frames

##################################################
#
#  Get parameters from command line
#
form Enter Info
  word sound_file
  real start_point 0.0
  real end_point 0.0
  real min_pitch
  real max_pitch
endform

##################################################
#
#  open sound file and extract portion specified
#
Open long sound file... 'sound_file$'
Extract part... 'start_point' 'end_point' no
Rename... sound

##################################################
#
#  Initialize variables
#
vcd2tot_frames    = undefined
min_f0            = undefined
max_f0            = undefined
median_f0         = undefined
mean_f0           = undefined
stdv_f0           = undefined
mas_f0            = undefined
min_eng           = undefined
max_eng           = undefined
mean_eng          = undefined
stdv_eng          = undefined

##################################################
#
#  Extract acoustic info from sound
#

# get the start, ending, and duration time of the sound
select Sound sound
start = Get starting time
end = Get finishing time
dur = end - start


if dur > (6.4 / 'min_pitch')
	# Query current F0 information
	select Sound sound
	To Pitch... 0 'min_pitch' 'max_pitch'
	vcd_frames = Count voiced frames
	tot_frames = Get number of frames
	vcd2tot_frames = vcd_frames / tot_frames
	min_f0 = Get minimum... 0 0 Hertz Parabolic
	# min_f0_time = Get time of minimum... 0 0 Hertz Parabolic
	# min_f0_loc = min_f0_time / dur
	max_f0 = Get maximum... 0 0 Hertz Parabolic
	# max_f0_time = Get time of maximum... 0 0 Hertz Parabolic
	# max_f0_loc = max_f0_time / dur
	median_f0 = Get quantile... 0 0 0.5 Hertz
	mean_f0 = Get mean... 0 0 Hertz
	stdv_f0 = Get standard deviation... 0 0 Hertz
	mas_f0 = Get mean absolute slope... Hertz

	# clean up
	select Pitch sound
	Remove

else
	vcd_frames = undefined
	tot_frames = undefined
	vcd2tot_frames = undefined
	min_f0 = undefined
	max_f0 = undefined
	median_f0 = undefined
	mean_f0 = undefined
	stdv_f0 = undefined
	mas_f0 = undefined

endif

# Query current intensity information if sound length is longer
# than that special praat ratio.  Otherwise fill with undefined
# values.
if dur > (6.4 / 'min_pitch')
     select Sound sound
     To Intensity... 'min_pitch' 0
     min_eng = Get minimum... 0 0 Parabolic
     # min_eng_time = Get time of minimum... 0 0 Parabolic
     # min_eng_loc = min_eng_time / dur
     max_eng = Get maximum... 0 0 Parabolic
     # max_eng_time = Get time of maximum... 0 0 Parabolic
     # max_eng_loc = max_eng_time / dur
     mean_eng = Get mean... 0 0
     stdv_eng = Get standard deviation... 0 0

     # clean up
     select Intensity sound
     Remove

else
     min_eng = undefined
     # min_eng_time = undefined
     # min_eng_loc = undefined
     max_eng = undefined
     # max_eng_time = undefined
     # max_eng_loc = undefined
     mean_eng = undefined
     stdv_eng = undefined

endif

# print the feature values
printline SECONDS:'dur:3'
printline F0_MAX:'max_f0:3'
printline F0_MIN:'min_f0:3'
printline F0_MEAN:'mean_f0:3'
printline F0_MEDIAN:'median_f0:3'
printline F0_STDV:'stdv_f0:3'
printline F0_MAS:'mas_f0:3'
printline ENG_MAX:'max_eng:3'
printline ENG_MIN:'min_eng:3'
printline ENG_MEAN:'mean_eng:3'
printline ENG_STDV:'stdv_eng:3'
printline VCD2TOT_FRAMES:'vcd2tot_frames:3'

# clean up
select Sound sound
Remove
