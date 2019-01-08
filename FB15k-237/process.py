#coding:utf-8
import sys

entity = set()
rel = set()

for line in open(sys.argv[1]):
	line = line.strip()
	if len(line) == 0:
		continue
	h, r, t = line.split('\t')
	entity.add(h)
	entity.add(t)
	rel.add(r)
for line in open(sys.argv[2]):
	line = line.strip()
	if len(line) == 0:
		continue
	h, r, t = line.split('\t')
	entity.add(h)
	entity.add(t)
	rel.add(r)

for line in open(sys.argv[3]):
	line = line.strip()
	if len(line) == 0:
		continue
	h, r, t = line.split('\t')
	entity.add(h)
	entity.add(t)
	rel.add(r)

e = list(entity)
r = list(rel)
e.sort()
r.sort()

for i in range(len(r)):
	print(r[i])
