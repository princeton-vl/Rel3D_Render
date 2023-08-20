wgetgdrive(){
  # $1 = file ID
  # $2 = file name

  URL="https://docs.google.com/uc?export=download&id=$1"

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

mkdir tmp
key="$1"
case $key in
	shapes)
		wgetgdrive 1hOp73YNCI9B1DqUHpf-1liWaHsNRiHzR tmp/shapes.zip
		unzip -o tmp/shapes.zip
		mkdir -p data
		rm -rf data/shapes
		mv shapes data
    		;;
	data_raw)
		wgetgdrive 1qJCrW2iTQiVtKQFiAjVsKP2bci4v941z tmp/20200223_selected.zip
		unzip -o tmp/20200223_selected.zip
		mkdir -p data
		mv 20200223 data
    		;;
    	*)
    		echo "unknow argument $1" # unknown argument
    		;;
esac
rm -r tmp
