import pickle as pkl

def main():
  f = open('training_data1000.pkl', 'rb')
  data = pkl.load(f)
  f.close()

  with open('data1000.json', 'w', encoding='utf-8') as fw:
    fw.write('[\n')
    for instr, inp, out in zip( data['instruction'], data['input'], data['output'] ):
      fw.write('\t{\n')
      fw.write(f'\t\t"instruction": "{instr}",\n')
      fw.write(f'\t\t"input": "{inp}",\n')
      fw.write(f'\t\t"output": "{out.split("::")[1][1:]}"\n')
      fw.write('\t},\n')
    fw.write(']\n')
  pass

if __name__ == '__main__':
  main()
