echo "Computing FID..."

python -m pytorch_fid ./imgs_fid/real ./imgs_fid/gen > log/fid.txt