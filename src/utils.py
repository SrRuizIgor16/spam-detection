import os

def sort_models(models, ascending=True):
    return {k: v for k, v in sorted(models.items(), key=lambda item: item[1]['Accuracy'], reverse=ascending)}

def save_plot(plt, path, name='plot'):
	if not os.path.exists(path):
		os.makedirs(path)
	path += f'/{name}.png'
	try:
		plt.savefig(path)
		print(f'Plot "{name}" saved successfully in {path} folder.')
	except Exception as e:
		print(f'Error saving plot:', e)
	plt.close()