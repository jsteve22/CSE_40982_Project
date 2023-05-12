import numpy as np
import pandas as pd
import pickle as pkl

def main():
  '''
  outputs = convert_sample_to_tuples(f'./samples/1337.txt')
  for o in outputs:
    for t in o:
      print(f'{t}\t',end=':\t')
    print()
    print()
  return
  '''
  with open('training_data100.pkl', 'wb') as fb:
    training = {}
    training['instruction'] = []
    training['input']       = []
    training['output']      = []

    for i in range(1,101):
      outputs = convert_sample_to_tuples(f'./samples/{i}.txt')
      for o in outputs:
        training['instruction'].append(o[0])
        training['input'].append(o[1])
        training['output'].append(o[2])
    pkl.dump(training, fb)
  pass

def convert_sample_to_tuples(filename):
  # instruction, input, output

  outputs = []

  with open(f'{filename}', 'r') as f:
    lines = f.readlines()

    q = 3
    instr = 'Ask user if they want to watch a popular movie'
    inp   = clean_strings(lines[:q-1])
    inp   = ''
    out   = lines[q].rstrip().split('::')[1][1:]
    outputs.append( (instr, inp, out) )

    q = 6
    instr = 'Ask user if they want to watch an old movie'
    inp   = clean_strings(lines[:q-1])
    inp   = ''
    out   = lines[q].rstrip().split('::')[1][1:]
    outputs.append( (instr, inp, out) )

    q = 9
    instr = 'Ask user if they want to watch a long movie'
    inp   = clean_strings(lines[:q-1])
    inp   = ''
    out   = lines[q].rstrip().split('::')[1][1:]
    outputs.append( (instr, inp, out) )

    q = 12
    instr = 'Ask the user what they think about a movie'
    inp   = clean_strings(lines[:q-1])
    inp   = ''
    out   = lines[q].rstrip().split('::')[1][1:]
    outputs.append( (instr, inp, out) )

    q = 15
    instr = 'Ask the user what they think about a movie'
    inp   = clean_strings(lines[:q-1])
    inp   = ''
    out   = lines[q].rstrip().split('::')[1][1:]
    outputs.append( (instr, inp, out) )

    q = 18
    instr = 'Ask the user what they think about a movie'
    inp   = clean_strings(lines[:q-1])
    inp   = ''
    out   = lines[q].rstrip().split('::')[1][1:]
    outputs.append( (instr, inp, out) )


    for q in range(21, len(lines), 3):
      ques_resp = []
      for i in range(0, q, 3):
        ques = lines[i].split(' ')[0]
        ques_resp.append( ques )
        if (ques == 'QUESTION:SIMILAR::'):
          ques_resp.append( ' '.join(lines[i].split(' ')[1:]) )
        ques_resp.append( lines[i+1].split('::')[1][1:] )
      instr = 'Recommend the user a movie to watch'
      # inp   = clean_strings(lines[:q-1])
      # print(ques_resp)
      inp   = clean_strings(ques_resp)
      out   = lines[q].rstrip()
      outputs.append( (instr, inp, out) )

  return outputs

def clean_strings(strings):
  ret = ''
  for s in strings:
    ret += f'{s.rstrip()} '
  return ret

if __name__ == '__main__':
  main()
