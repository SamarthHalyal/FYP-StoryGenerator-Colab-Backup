./back.sh &
file_updated()
{
	testcommand=`diff $1 ./drive/My\ Drive/Colab\ Notebooks/$1 > check.txt`
	if [ -s check.txt ]; then
		echo "modified"
		cp ./drive/My\ Drive/Colab\ Notebooks/predictions.txt predictions.txt
		cp ./drive/My\ Drive/Colab\ Notebooks/input1.txt input1.txt
		python predict.py
		php ./project_zip/test.php
	fi
}

inotifywait -e modify -m . |
while read -r drive events filename; do
	#if [ "$filename" = "input1.txt" ]; then
	file_updated input1.txt
	#fi
done