import sys, csv, re, os
import pandas as pd
import numpy as np
import multiprocessing as mp
import concurrent.futures as con
import clean

time_log_pattern = re.compile('\[\d{4}/\d{2}/\d{2}-(?:\d{1,2}:){2}\d{1,2}\],\n')
def process(lines, core):
	print('core {} started'.format(core))
	tableData = pd.DataFrame(columns=['time','text'])
	tmp = ''
	i=0
	for line in lines:
		if bool(time_log_pattern.search(line)):
			result = clean.stripping(tmp.replace('\ufeff',''))
			result = clean.stripping(clean.fixing(clean.filtering(result)))
			if len(result) == 0:
				tmp = ''
				continue
			tableData = tableData.append({
				'time': line[1:-3],
				'text': result
			}, ignore_index=True)
			tmp = ''
			i+=1
		else:
			tmp += clean.cleanLine(line)
	print('core {} finished'.format(core))
	return (core, tableData)

if __name__ == '__main__':
	try:
		file = open(sys.argv[1], 'r', encoding='utf8')
		if os.name == 'nt':
			fileName = sys.argv[1].split('\\')[-1].split('.')[0]
		else:
			fileName = sys.argv[1].split('/')[-1].split('.')[0]
	except IndexError:
		print('Please give the input file')
		exit(0)
	except FileNotFoundError:
		print('File not found')
		exit(0)

	ncore = mp.cpu_count()

	all_line = file.readlines()
	line_chunks = np.array_split(all_line, ncore)

	output_list = []
	with con.ProcessPoolExecutor(max_workers=ncore) as executor:
		workers = {executor.submit(process, line_chunks[core], core): core for core in range(ncore)}
		for worker in con.as_completed(workers):
			result_tmp = workers[worker]
			try:
				data = worker.result()
			except Exception as exc:
				print('Error occurred : {}'.format(exc))
			else:
				output_list.append(data)

	results_tup = sorted(output_list, key=lambda x: x[0])
	result_table = pd.concat(list(map(lambda data: data[1], results_tup)))

	result_table.to_csv('data/clean/'+fileName+'_clean.csv', index=False,
					quoting=csv.QUOTE_NONNUMERIC, doublequote=False, escapechar="\\")