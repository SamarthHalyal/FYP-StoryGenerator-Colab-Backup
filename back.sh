curtime=`date +%s`
while :
do
	temp=$((curtime + 3))
	if [ `date +%s` -ge $temp ]; then
		curtime=$(( $curtime + 3 ))
		python update.py
		cat input1.txt > temp.txt
	fi
done